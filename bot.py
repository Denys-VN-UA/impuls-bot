# bot.py
# IMPULS ⚡ FINAL v3.2 — TwelveData
# ✅ Dynamic RSI (more signals without spam)
# ✅ Optional higher-trend filter
# ✅ Trading schedule (Mon–Fri, 10:00–20:00 by TZ)
# ✅ TOP-N mode, cooldown per pair
# ✅ Auto expiry report (no ID in text)
# ✅ Owner commands: /test /stats /report_now /pulse_on /pulse_off /whoami /debug_pairs
#
# Requirements:
#   python-telegram-bot[job-queue]==22.5
#   requests, pandas, numpy

import os
import logging
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Optional, Any, List, Dict, Tuple

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes


# =========================
# ENV / SETTINGS
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "").strip()
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "IMPULS ⚡")

TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Kyiv").strip()
TZ = ZoneInfo(TIMEZONE_NAME)

# Trading schedule (Mon–Fri 10:00–20:00)
TRADE_START = os.getenv("TRADE_START", "10:00").strip()  # HH:MM
TRADE_END = os.getenv("TRADE_END", "20:00").strip()      # HH:MM

# Scanner
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "300"))  # 5 min default
TF = os.getenv("TF", "1min").strip()
CANDLES = int(os.getenv("CANDLES", "250"))

# Expiry
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "3"))

# Filters
MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", "52"))
ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.006"))  # ATR% threshold (in percent)
ADAPTIVE_FILTERS = os.getenv("ADAPTIVE_FILTERS", "1").strip().lower() in ("1", "true", "yes", "y")
GLOBAL_ATR_MULT = float(os.getenv("GLOBAL_ATR_MULT", "0.95"))

# Higher trend filter
TREND_FILTER = os.getenv("TREND_FILTER", "1").strip().lower() in ("1", "true", "yes", "y")
TREND_TF = os.getenv("TREND_TF", "15min").strip()

# Sending mode
SEND_MODE = os.getenv("SEND_MODE", "TOP").strip().upper()  # TOP / BEST / ALL
TOP_N = int(os.getenv("TOP_N", "1"))

# Cooldown per symbol
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "7"))

# Anti-spam messages
WEAK_MSG_COOLDOWN_MINUTES = int(os.getenv("WEAK_MSG_COOLDOWN_MINUTES", "60"))
OFFTIME_MSG_COOLDOWN_MINUTES = int(os.getenv("OFFTIME_MSG_COOLDOWN_MINUTES", "120"))
APILIMIT_MSG_COOLDOWN_MINUTES = int(os.getenv("APILIMIT_MSG_COOLDOWN_MINUTES", "120"))

# Pulse
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "1800"))  # 30 min
PULSE_ENABLED = os.getenv("PULSE_ENABLED", "1").strip().lower() in ("1", "true", "yes", "y")

# Daily report time
REPORT_HOUR = int(os.getenv("REPORT_HOUR", "20"))
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", "5"))


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("impuls")


# =========================
# SYMBOLS parsing (important fix)
# =========================
def parse_symbols(raw: str) -> List[str]:
    # Accept commas / pipes / semicolons / newlines
    s = (raw or "").strip()
    s = s.replace("|", ",").replace(";", ",").replace("\n", ",").replace("\r", ",")
    items = [x.strip() for x in s.split(",") if x.strip()]
    # Remove accidental duplicates
    out = []
    seen = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

SYMBOLS = parse_symbols(os.getenv("SYMBOLS", "EUR/USD,GBP/USD,USD/JPY,USD/CHF,EUR/JPY,GBP/JPY"))


# =========================
# STATS (in-memory)
# =========================
STATS: Dict[str, Any] = {
    "day": None,
    "signals": 0,
    "win": 0,
    "loss": 0,
    "last_signal": None,
    "pulse_on": True,
    "cooldown": {},

    "last_weak_msg": None,
    "last_oftime_msg": None,
    "last_api_msg": None,
}


# =========================
# TIME UTILS
# =========================
def now_tz() -> datetime:
    return datetime.now(TZ)

def today_key() -> str:
    return now_tz().strftime("%Y-%m-%d")

def ensure_day_reset() -> None:
    d = today_key()
    if STATS["day"] != d:
        STATS["day"] = d
        STATS["signals"] = 0
        STATS["win"] = 0
        STATS["loss"] = 0
        STATS["last_signal"] = None
        STATS["cooldown"] = {}
        log.info("Daily stats reset for %s (%s)", d, TIMEZONE_NAME)

def fmt_time(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%H:%M:%S")

def parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(hour=int(hh), minute=int(mm), tzinfo=TZ)

def is_trading_time(dt: datetime) -> bool:
    wd = dt.weekday()  # Mon=0 ... Sun=6
    if wd >= 5:
        return False
    start_t = parse_hhmm(TRADE_START)
    end_t = parse_hhmm(TRADE_END)
    t = dt.timetz()
    return (t >= start_t) and (t <= end_t)

def minutes_ago(ts_iso: Optional[str]) -> Optional[float]:
    if not ts_iso:
        return None
    try:
        dt = datetime.fromisoformat(ts_iso)
        return (now_tz() - dt).total_seconds() / 60.0
    except Exception:
        return None

def can_send_throttled(key: str, cooldown_minutes: int) -> bool:
    last_iso = STATS.get(key)
    ago = minutes_ago(last_iso)
    if ago is None or ago >= cooldown_minutes:
        STATS[key] = now_tz().isoformat()
        return True
    return False


# =========================
# REQUIRE ENV
# =========================
def require_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN пустой. Добавь BOT_TOKEN в Railway Variables.")
    if not TWELVE_API_KEY:
        raise RuntimeError("TWELVE_API_KEY пустой. Добавь TWELVE_API_KEY в Railway Variables.")
    if not CHANNEL_ID:
        raise RuntimeError("CHANNEL_ID пустой. Добавь CHANNEL_ID в Railway Variables.")
    if OWNER_ID == 0:
        log.warning("OWNER_ID не задан (0). Owner-команды и WIN/LOSS будут недоступны.")


# =========================
# TWELVEDATA API
# =========================
TD_BASE = "https://api.twelvedata.com"

def td_get(path: str, params: dict, timeout: int = 20) -> dict:
    url = f"{TD_BASE}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"TwelveData: bad response (HTTP {r.status_code})")

    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(data.get("message") or "TwelveData error")
    return data

def td_time_series(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    data = td_get(
        "/time_series",
        {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": TWELVE_API_KEY,
            "format": "JSON",
            "timezone": "UTC",
        },
        timeout=25,
    )

    values = (data or {}).get("values") or []
    if not values:
        raise RuntimeError("No candles returned")

    df = pd.DataFrame(values)
    df = df.iloc[::-1].reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df

def td_price(symbol: str) -> float:
    data = td_get(
        "/price",
        {"symbol": symbol, "apikey": TWELVE_API_KEY, "format": "JSON"},
        timeout=15,
    )
    p = data.get("price")
    if p is None:
        raise RuntimeError("No price returned")
    return float(p)


# =========================
# INDICATORS
# =========================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    a = atr_series(df, period).iloc[-1]
    c = df["close"].iloc[-1]
    if c == 0 or pd.isna(a) or pd.isna(c):
        return 0.0
    return float((a / c) * 100.0)

def adaptive_atr_threshold(df: pd.DataFrame) -> float:
    base = max(0.0, ATR_THRESHOLD)
    if not ADAPTIVE_FILTERS:
        return base * GLOBAL_ATR_MULT

    try:
        a = atr_series(df, 14)
        c = df["close"]
        atrp = (a / c) * 100.0
        tail = atrp.dropna().tail(60)
        if len(tail) < 20:
            return base * GLOBAL_ATR_MULT
        med = float(tail.median())
        thr = max(base, 0.80 * med) * GLOBAL_ATR_MULT
        return float(thr)
    except Exception:
        return base * GLOBAL_ATR_MULT


# =========================
# V3.2: Dynamic RSI bands + probability mapping
# =========================
def rsi_bands(direction: str, atrp: float, thr: float) -> Tuple[float, float]:
    """
    Dynamic RSI bands:
      - Wider when volatility is healthy (more signals)
      - Tighter when volatility is weak (avoid noise)
    """
    if direction.upper() == "CALL":
        lo, hi = 43.0, 67.0
    else:
        lo, hi = 33.0, 57.0

    rel = atrp / max(thr, 0.0001)

    # Expand if market is "alive"
    if rel >= 1.6:
        lo -= 4.0
        hi += 4.0
    elif rel >= 1.2:
        lo -= 2.0
        hi += 2.0

    # Tighten if weak
    if rel < 1.0:
        lo += 2.0
        hi -= 2.0

    lo = max(10.0, lo)
    hi = min(90.0, hi)
    return lo, hi

def prob_from_score(score: int) -> int:
    return max(50, min(92, int(score)))


# =========================
# SIGNAL LOGIC
# =========================
@dataclass
class Signal:
    symbol: str
    direction: str  # CALL/PUT
    probability: int
    price: float
    rsi14: float
    ema50: float
    ema200: float
    atr14_pct: float
    entry_time: datetime
    exit_time: datetime
    entry_price: float


def trend_ok(symbol: str) -> Optional[str]:
    """
    Higher trend (TREND_TF):
      CALL if EMA50 > EMA200
      PUT  if EMA50 < EMA200
    """
    if not TREND_FILTER:
        return None
    df = td_time_series(symbol, TREND_TF, 220)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    e50 = float(df["ema50"].iloc[-1])
    e200 = float(df["ema200"].iloc[-1])
    if e50 > e200:
        return "CALL"
    if e50 < e200:
        return "PUT"
    return None


def compute_signal(symbol: str) -> Optional[Signal]:
    df = td_time_series(symbol, TF, CANDLES)

    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)

    close = float(df["close"].iloc[-1])
    ema50_v = float(df["ema50"].iloc[-1])
    ema200_v = float(df["ema200"].iloc[-1])
    rsi_v = float(df["rsi14"].iloc[-1])
    atrp = atr_percent(df, 14)

    thr = adaptive_atr_threshold(df)
    if atrp < thr:
        return None

    # Local EMA trend decides direction
    trend_up = ema50_v > ema200_v
    trend_down = ema50_v < ema200_v

    if trend_up:
        direction = "CALL"
        score = 35
    elif trend_down:
        direction = "PUT"
        score = 35
    else:
        return None

    # Dynamic RSI check
    lo, hi = rsi_bands(direction, atrp, thr)
    if not (lo <= rsi_v <= hi):
        return None
    score += 30

    # Higher-trend filter: bonus if matches, reject if opposite
    if TREND_FILTER:
        tdir = trend_ok(symbol)
        if tdir is not None and tdir != direction:
            return None
        if tdir == direction:
            score += 10

    # Volatility bonus (cap)
    rel = atrp / max(thr, 0.0001)
    score += min(20, int(rel * 6))

    probability = prob_from_score(score)
    if probability < MIN_PROBABILITY:
        return None

    entry = now_tz()
    exit_ = entry + timedelta(minutes=EXPIRY_MINUTES)

    return Signal(
        symbol=symbol,
        direction=direction,
        probability=probability,
        price=close,
        rsi14=rsi_v,
        ema50=ema50_v,
        ema200=ema200_v,
        atr14_pct=atrp,
        entry_time=entry,
        exit_time=exit_,
        entry_price=close,
    )


def in_cooldown(symbol: str) -> bool:
    ts = STATS["cooldown"].get(symbol)
    if not ts:
        return False
    ago = minutes_ago(ts)
    return (ago is not None) and (ago < COOLDOWN_MINUTES)

def mark_cooldown(symbol: str) -> None:
    STATS["cooldown"][symbol] = now_tz().isoformat()

def pick_signals(symbols: List[str]) -> List[Signal]:
    found: List[Signal] = []
    for s in symbols:
        if in_cooldown(s):
            continue
        sig = compute_signal(s)
        if sig:
            found.append(sig)

    found.sort(key=lambda x: x.probability, reverse=True)
    if not found:
        return []

    if SEND_MODE == "ALL":
        return found
    if SEND_MODE == "BEST":
        return found[:1]
    return found[:max(1, min(3, TOP_N))]


# =========================
# TELEGRAM MESSAGES
# =========================
def direction_label(direction: str) -> str:
    return "⬆️ ВВЕРХ" if direction.upper() == "CALL" else "⬇️ ВНИЗ"

def sign_dir_from_prices(entry: float, last: float) -> str:
    if last > entry:
        return "⬆️ ВВЕРХ"
    if last < entry:
        return "⬇️ ВНИЗ"
    return "➡️ РОВНО"

def pct_change(entry: float, last: float) -> float:
    if entry == 0:
        return 0.0
    return ((last - entry) / entry) * 100.0

def winloss_keyboard(signal_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ WIN", callback_data=f"wl|win|{signal_id}"),
        InlineKeyboardButton("❌ LOSS", callback_data=f"wl|loss|{signal_id}"),
    ]])

def signal_message(sig: Signal) -> str:
    return (
        f"📊 *СИГНАЛ {sig.symbol}*  📈\n"
        f"🎯 Направление: *{direction_label(sig.direction)}*\n"
        f"🔥 Вероятность: *{sig.probability}%*\n\n"
        f"💰 Цена: `{sig.price:.5f}`\n"
        f"⚡ ATR(14): `{sig.atr14_pct:.3f}%`\n"
        f"📉 RSI(14): `{sig.rsi14:.1f}`\n"
        f"📍 EMA50: `{sig.ema50:.5f}`\n"
        f"📍 EMA200: `{sig.ema200:.5f}`\n\n"
        f"⏱ Вход: *{fmt_time(sig.entry_time)}*\n"
        f"🏁 Выход: *{fmt_time(sig.exit_time)}*  (эксп. {EXPIRY_MINUTES} мин)\n"
        f"🌍 Таймзона: `{TIMEZONE_NAME}`"
    )

def offtime_message() -> str:
    return (
        f"🌙 Сейчас не торговое время.\n"
        f"📅 Торгую ПН–ПТ\n"
        f"⏰ {TRADE_START}–{TRADE_END} ({TIMEZONE_NAME})"
    )

def weak_market_message() -> str:
    return "📉 Рынок слабый — сильных сигналов нет.\nЯ продолжаю анализировать…"

def api_limit_message() -> str:
    return (
        "⚠️ Данные временно недоступны (лимит API).\n"
        "Я не молчу — провайдер ограничил запросы.\n"
        "Уменьши частоту/список пар или подожди."
    )

async def post_to_channel(context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None) -> None:
    await context.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )


# =========================
# AUTO EXPIRY REPORT (no ID)
# =========================
async def job_expiry_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data or {}
    symbol = data["symbol"]
    direction = data["direction"]
    entry_price = float(data["entry_price"])

    try:
        last_price = td_price(symbol)
    except Exception as e:
        log.warning("Expiry price fetch failed for %s: %s", symbol, e)
        return

    move_label = sign_dir_from_prices(entry_price, last_price)
    delta = pct_change(entry_price, last_price)
    delta_str = f"{delta:+.3f}%"

    quote_win = (last_price > entry_price) if direction.upper() == "CALL" else (last_price < entry_price)
    quote_result = "✅ WIN" if quote_win else "❌ LOSS"

    text = (
        f"⏱ Экспирация прошла по *{symbol}*\n"
        f"📈 График пошёл: *{move_label}*\n"
        f"💰 Цена: `{entry_price:.5f}` → `{last_price:.5f}`  ({delta_str})\n"
        f"✅ По котировкам это *{quote_result}*\n\n"
        f"👉 Если у Pocket Option итог отличается — отметь вручную WIN/LOSS под сигналом."
    )
    await post_to_channel(context, text)


# =========================
# JOBS
# =========================
async def job_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_day_reset()
    now = now_tz()

    if not is_trading_time(now):
        if can_send_throttled("last_oftime_msg", OFFTIME_MSG_COOLDOWN_MINUTES):
            await post_to_channel(context, offtime_message())
        return

    try:
        signals = pick_signals(SYMBOLS)
    except Exception as e:
        msg = str(e).lower()
        if ("credit" in msg) or ("limit" in msg) or ("429" in msg) or ("too many" in msg):
            if can_send_throttled("last_api_msg", APILIMIT_MSG_COOLDOWN_MINUTES):
                await post_to_channel(context, api_limit_message())
            return
        log.exception("Signal scan error: %s", e)
        return

    if not signals:
        if can_send_throttled("last_weak_msg", WEAK_MSG_COOLDOWN_MINUTES):
            await post_to_channel(context, weak_market_message())
        return

    for sig in signals:
        STATS["signals"] += 1
        mark_cooldown(sig.symbol)

        signal_id = f"{sig.entry_time.strftime('%Y%m%d%H%M%S')}_{sig.symbol.replace('/', '')}"
        STATS["last_signal"] = {"symbol": sig.symbol, "time": sig.entry_time.isoformat(), "prob": sig.probability}

        await post_to_channel(context, signal_message(sig), reply_markup=winloss_keyboard(signal_id))

        delay = EXPIRY_MINUTES * 60 + 2
        context.job_queue.run_once(
            job_expiry_report,
            when=delay,
            data={"symbol": sig.symbol, "direction": sig.direction, "entry_price": sig.entry_price},
            name=f"expiry_{signal_id}",
        )

async def job_pulse(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not PULSE_ENABLED:
        return
    if not STATS.get("pulse_on", True):
        return
    if not is_trading_time(now_tz()):
        return
    await post_to_channel(context, f"🕒 *{CHANNEL_NAME}*: бот жив, анализирую рынок…")

async def job_daily_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_day_reset()
    s = STATS["signals"]
    w = STATS["win"]
    l = STATS["loss"]
    wr = (w / max(1, w + l)) * 100.0

    txt = (
        f"📌 *{CHANNEL_NAME} — ЕЖЕДНЕВНЫЙ ОТЧЁТ*\n"
        f"🗓 Дата: *{now_tz().strftime('%d.%m.%Y')}*  ({TIMEZONE_NAME})\n\n"
        f"✉️ Сигналов: *{s}*\n"
        f"✅ WIN: *{w}*\n"
        f"❌ LOSS: *{l}*\n"
        f"🎯 WinRate: *{wr:.1f}%*"
    )
    await post_to_channel(context, txt)


# =========================
# COMMANDS / HANDLERS
# =========================
def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ IMPULS запущен.\n\n"
        f"Канал: {CHANNEL_NAME}\n"
        f"Таймзона: {TIMEZONE_NAME}\n"
        f"Торговля: ПН–ПТ {TRADE_START}–{TRADE_END}\n"
        f"TF: {TF} | Expiry: {EXPIRY_MINUTES}m\n"
        f"Symbols: {', '.join(SYMBOLS)}\n\n"
        "Owner команды:\n"
        "/test\n/stats\n/report_now\n/pulse_on\n/pulse_off\n/whoami\n/debug_pairs\n",
        disable_web_page_preview=True,
    )

async def whoami_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    await update.message.reply_text(
        f"👤 You: {u.full_name}\n"
        f"🆔 user_id: {u.id}\n"
        f"✅ owner: {'YES' if is_owner(u.id) else 'NO'}\n"
        f"OWNER_ID env: {OWNER_ID}"
    )

async def test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    await post_to_channel(context, "✅ *ТЕСТ*: бот может писать в канал (OK)")
    await update.message.reply_text("✅ Тест отправлен в канал.")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    ensure_day_reset()
    s = STATS["signals"]
    w = STATS["win"]
    l = STATS["loss"]
    wr = (w / max(1, w + l)) * 100.0
    last = STATS.get("last_signal")
    await update.message.reply_text(
        f"📊 Статистика (за сегодня)\n"
        f"Сигналов: {s}\n"
        f"WIN: {w}\n"
        f"LOSS: {l}\n"
        f"WinRate: {wr:.1f}%\n"
        f"Последний: {last if last else '—'}"
    )

async def report_now_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    await job_daily_report(context)
    await update.message.reply_text("✅ Отчёт отправлен в канал.")

async def pulse_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    STATS["pulse_on"] = True
    await update.message.reply_text("✅ Пульс включён.")

async def pulse_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    STATS["pulse_on"] = False
    await update.message.reply_text("✅ Пульс выключён.")

async def debug_pairs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Owner-only: shows why pairs PASS/REJECT (like your log).
    """
    if not is_owner(update.effective_user.id):
        return

    lines = []
    for sym in SYMBOLS:
        # cooldown check
        cd = "CD ✅" if in_cooldown(sym) else "CD —"
        try:
            df = td_time_series(sym, TF, CANDLES)
            df["ema50"] = ema(df["close"], 50)
            df["ema200"] = ema(df["close"], 200)
            df["rsi14"] = rsi(df["close"], 14)

            close = float(df["close"].iloc[-1])
            ema50_v = float(df["ema50"].iloc[-1])
            ema200_v = float(df["ema200"].iloc[-1])
            rsi_v = float(df["rsi14"].iloc[-1])
            atrp = atr_percent(df, 14)
            thr = adaptive_atr_threshold(df)

            if atrp < thr:
                lines.append(f"{sym:10s} | REJECT ❌ | {cd} | ATR={atrp:.3f}% thr={thr:.3f}% | weak ATR")
                continue

            if ema50_v > ema200_v:
                cand = "CALL"
                trend = "UP"
            elif ema50_v < ema200_v:
                cand = "PUT"
                trend = "DOWN"
            else:
                lines.append(f"{sym:10s} | REJECT ❌ | {cd} | EMA flat")
                continue

            lo, hi = rsi_bands(cand, atrp, thr)
            if not (lo <= rsi_v <= hi):
                lines.append(
                    f"{sym:10s} | REJECT ❌ | {cd} | cand={cand:<4s} | ATR={atrp:.3f}% thr={thr:.3f}% | "
                    f"RSI={rsi_v:.1f} band={lo:.1f}-{hi:.1f} | EMAtrend={trend} | RSI out"
                )
                continue

            # higher trend check
            if TREND_FILTER:
                try:
                    tdir = trend_ok(sym)
                    if tdir is not None and tdir != cand:
                        lines.append(f"{sym:10s} | REJECT ❌ | {cd} | cand={cand} | higher trend mismatch")
                        continue
                except Exception as e:
                    # If higher trend fails, don't block; just note
                    pass

            # score/prob estimate (same as compute_signal)
            score = 35 + 30
            rel = atrp / max(thr, 0.0001)
            score += min(20, int(rel * 6))
            if TREND_FILTER:
                # we can't be sure tdir==cand here; keep neutral
                pass
            prob = prob_from_score(score)

            verdict = "PASS ✅" if prob >= MIN_PROBABILITY and (not in_cooldown(sym)) else "REJECT ❌"
            lines.append(
                f"{sym:10s} | {verdict:8s} | {cd} | cand={cand:<4s} | prob={prob}% | "
                f"ATR={atrp:.3f}% thr={thr:.3f}% | RSI={rsi_v:.1f} band={lo:.1f}-{hi:.1f} | EMAtrend={trend}"
            )

        except Exception as e:
            lines.append(f"{sym:10s} | ERROR ❌ | {str(e)[:120]}")

    text = "```text\n" + "\n".join(lines[:60]) + "\n```"
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()

    if not is_owner(q.from_user.id):
        await q.answer("⛔ Только владелец может отмечать WIN/LOSS.", show_alert=True)
        return

    parts = (q.data or "").split("|")
    if len(parts) != 3 or parts[0] != "wl":
        return

    ensure_day_reset()
    action = parts[1]

    if action == "win":
        STATS["win"] += 1
        await q.message.reply_text("✅ WIN отмечен")
    elif action == "loss":
        STATS["loss"] += 1
        await q.message.reply_text("❌ LOSS отмечен")


# =========================
# MAIN
# =========================
def main() -> None:
    require_env()
    ensure_day_reset()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("whoami", whoami_cmd))
    app.add_handler(CommandHandler("test", test_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("report_now", report_now_cmd))
    app.add_handler(CommandHandler("pulse_on", pulse_on_cmd))
    app.add_handler(CommandHandler("pulse_off", pulse_off_cmd))
    app.add_handler(CommandHandler("debug_pairs", debug_pairs_cmd))
    app.add_handler(CallbackQueryHandler(on_button))

    if app.job_queue is None:
        raise RuntimeError("JobQueue не активен. Установи python-telegram-bot[job-queue]==22.5")

    app.job_queue.run_repeating(job_signals, interval=SIGNAL_INTERVAL_SECONDS, first=10, name="signals")
    app.job_queue.run_repeating(job_pulse, interval=PULSE_INTERVAL_SECONDS, first=60, name="pulse")

    report_t = time(hour=REPORT_HOUR, minute=REPORT_MINUTE, tzinfo=TZ)
    app.job_queue.run_daily(job_daily_report, time=report_t, name="daily_report")

    log.info(
        "IMPULS v3.2 started | TZ=%s | Trade=%s-%s | Symbols=%s | Mode=%s TOP_N=%s | TF=%s | Exp=%sm | "
        "ATR=%s adaptive=%s mult=%s | Trend=%s TrendTF=%s | MIN_PROB=%s",
        TIMEZONE_NAME, TRADE_START, TRADE_END, SYMBOLS, SEND_MODE, TOP_N, TF, EXPIRY_MINUTES,
        ATR_THRESHOLD, ADAPTIVE_FILTERS, GLOBAL_ATR_MULT, TREND_FILTER, TREND_TF, MIN_PROBABILITY
    )

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
