# bot.py
# IMPULS ⚡ FINAL v3.1 — TwelveData, TOP-N, no-spam weak market, trading schedule,
# auto expiry report (no ID shown), + /diag diagnostics + /whoami
# python-telegram-bot[job-queue]==22.5

import os
import logging
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Optional, Any, List, Dict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes


# =========================
# ENV / НАСТРОЙКИ
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "").strip()
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "IMPULS ⚡")

TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Kyiv")
TZ = ZoneInfo(TIMEZONE_NAME)

# Торговые дни/время (ПН–ПТ 10:00–20:00)
TRADE_START = os.getenv("TRADE_START", "10:00").strip()  # HH:MM
TRADE_END = os.getenv("TRADE_END", "20:00").strip()      # HH:MM

# Сканер
SYMBOLS = [x.strip() for x in os.getenv("SYMBOLS", "EUR/USD,USD/JPY,USD/CHF").split(",") if x.strip()]
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "600"))
TF = os.getenv("TF", "1min").strip()
CANDLES = int(os.getenv("CANDLES", "250"))

# Экспирация (минуты)
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "3"))

# Фильтры
MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", "58"))
ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.012"))  # ATR% порог (в процентах)
ADAPTIVE_FILTERS = os.getenv("ADAPTIVE_FILTERS", "1").strip() in ("1", "true", "True", "YES", "yes")
GLOBAL_ATR_MULT = float(os.getenv("GLOBAL_ATR_MULT", "1.00"))

# Старший тренд
TREND_FILTER = os.getenv("TREND_FILTER", "1").strip() in ("1", "true", "True", "YES", "yes")
TREND_TF = os.getenv("TREND_TF", "15min").strip()

# Отправка сигналов
SEND_MODE = os.getenv("SEND_MODE", "TOP").strip().upper()  # TOP / BEST / ALL
TOP_N = int(os.getenv("TOP_N", "2"))

# Кулдаун на одну пару
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "6"))

# Ограничители спама “рынок слабый / не торговое время / лимит API”
WEAK_MSG_COOLDOWN_MINUTES = int(os.getenv("WEAK_MSG_COOLDOWN_MINUTES", "45"))
OFFTIME_MSG_COOLDOWN_MINUTES = int(os.getenv("OFFTIME_MSG_COOLDOWN_MINUTES", "60"))
APILIMIT_MSG_COOLDOWN_MINUTES = int(os.getenv("APILIMIT_MSG_COOLDOWN_MINUTES", "60"))

# Пульс (сообщение “бот жив”)
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "1800"))
PULSE_ENABLED = os.getenv("PULSE_ENABLED", "1").strip() in ("1", "true", "True", "YES", "yes")

# Ежедневный отчёт (по умолчанию 20:05 в таймзоне)
REPORT_HOUR = int(os.getenv("REPORT_HOUR", "20"))
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", "5"))


# =========================
# ЛОГИ
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("impuls")


# =========================
# СТАТИСТИКА (в памяти)
# =========================
STATS: Dict[str, Any] = {
    "day": None,  # YYYY-MM-DD (TZ)
    "signals": 0,
    "win": 0,
    "loss": 0,
    "last_signal": None,
    "pulse_on": True,
    "cooldown": {},  # symbol -> iso timestamp
    "last_weak_msg": None,
    "last_oftime_msg": None,
    "last_api_msg": None,
}


# =========================
# УТИЛИТЫ ВРЕМЕНИ
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
    wd = dt.weekday()  # ПН=0 ... ВС=6
    if wd >= 5:
        return False
    start_t = parse_hhmm(TRADE_START)
    end_t = parse_hhmm(TRADE_END)
    t = dt.timetz()
    return (t >= start_t) and (t <= end_t)

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
        log.warning("OWNER_ID не задан (0). Команды owner-only и WIN/LOSS недоступны.")


# =========================
# TWELVEDATA
# =========================
TD_BASE = "https://api.twelvedata.com"

def td_time_series(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    url = f"{TD_BASE}/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_API_KEY,
        "format": "JSON",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(data.get("message") or "TwelveData error")

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
    url = f"{TD_BASE}/price"
    params = {"symbol": symbol, "apikey": TWELVE_API_KEY, "format": "JSON"}
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(data.get("message") or "TwelveData price error")
    p = data.get("price")
    if p is None:
        raise RuntimeError("No price returned")
    return float(p)


# =========================
# ИНДИКАТОРЫ
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
    base = ATR_THRESHOLD
    if not ADAPTIVE_FILTERS:
        return max(0.0, base) * GLOBAL_ATR_MULT

    try:
        a = atr_series(df, 14)
        c = df["close"]
        atrp = (a / c) * 100.0
        tail = atrp.dropna().tail(60)
        if len(tail) < 20:
            return max(0.0, base) * GLOBAL_ATR_MULT
        med = float(tail.median())
        thr = max(base, 0.80 * med) * GLOBAL_ATR_MULT
        return float(thr)
    except Exception:
        return max(0.0, base) * GLOBAL_ATR_MULT


# =========================
# ЛОГИКА СИГНАЛА
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

    trend_up = ema50_v > ema200_v
    trend_down = ema50_v < ema200_v

    direction = None
    score = 0

    if trend_up:
        score += 35
        if 45 <= rsi_v <= 65:
            score += 35
            direction = "CALL"
    elif trend_down:
        score += 35
        if 35 <= rsi_v <= 55:
            score += 35
            direction = "PUT"
    else:
        return None

    if direction is None:
        return None

    if TREND_FILTER:
        tdir = trend_ok(symbol)
        if tdir is not None and tdir != direction:
            return None
        if tdir == direction:
            score += 12

    rel = atrp / max(thr, 0.0001)
    vol_bonus = min(20, int(rel * 6))
    score += vol_bonus

    probability = max(55, min(92, int(score)))
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
# TELEGRAM: сообщения
# =========================
def winloss_keyboard(signal_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ WIN", callback_data=f"wl|win|{signal_id}"),
            InlineKeyboardButton("❌ LOSS", callback_data=f"wl|loss|{signal_id}"),
        ]
    ])

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
        "Я не молчу — просто провайдер ограничил запросы.\n"
        "Попробуй позже или уменьши частоту/список пар."
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
# AUTO EXPIRY REPORT
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
# DIAG (V3.1)
# =========================
def _bool_emoji(v: bool) -> str:
    return "✅" if v else "❌"

def _trend_local(ema50_v: float, ema200_v: float) -> str:
    if ema50_v > ema200_v:
        return "UP"
    if ema50_v < ema200_v:
        return "DOWN"
    return "FLAT"

def diag_symbol(symbol: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"symbol": symbol}
    out["cooldown"] = in_cooldown(symbol)

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

    out.update({
        "price": close,
        "ema50": ema50_v,
        "ema200": ema200_v,
        "rsi": rsi_v,
        "atrp": atrp,
        "thr": thr,
        "local_trend": _trend_local(ema50_v, ema200_v),
    })

    # Правила, как в compute_signal
    reasons = []

    atr_ok = atrp >= thr
    if not atr_ok:
        reasons.append(f"ATR low ({atrp:.3f}% < {thr:.3f}%)")

    trend_up = ema50_v > ema200_v
    trend_down = ema50_v < ema200_v
    if not (trend_up or trend_down):
        reasons.append("EMA trend flat")

    # RSI окна
    candidate_dir = None
    rsi_ok = False
    if trend_up:
        candidate_dir = "CALL"
        rsi_ok = (45 <= rsi_v <= 65)
        if not rsi_ok:
            reasons.append(f"RSI out for CALL ({rsi_v:.1f} not in 45-65)")
    elif trend_down:
        candidate_dir = "PUT"
        rsi_ok = (35 <= rsi_v <= 55)
        if not rsi_ok:
            reasons.append(f"RSI out for PUT ({rsi_v:.1f} not in 35-55)")

    # Старший тренд
    trend_tf_ok = True
    tdir = None
    if TREND_FILTER:
        tdir = trend_ok(symbol)
        if tdir is not None and candidate_dir is not None and tdir != candidate_dir:
            trend_tf_ok = False
            reasons.append(f"TrendTF mismatch ({TREND_TF} says {tdir}, local {candidate_dir})")

    # Score/prob как в compute_signal (приблизительно)
    score = 0
    if trend_up or trend_down:
        score += 35
    if rsi_ok:
        score += 35
    if TREND_FILTER and tdir is not None and tdir == candidate_dir:
        score += 12
    if atr_ok:
        rel = atrp / max(thr, 0.0001)
        score += min(20, int(rel * 6))
    probability = max(55, min(92, int(score)))

    prob_ok = probability >= MIN_PROBABILITY
    if not prob_ok:
        reasons.append(f"prob low ({probability}% < {MIN_PROBABILITY}%)")

    pass_all = (not out["cooldown"]) and atr_ok and (trend_up or trend_down) and rsi_ok and trend_tf_ok and prob_ok

    out.update({
        "candidate": candidate_dir,
        "prob": probability,
        "pass": pass_all,
        "reasons": reasons,
    })
    return out


# =========================
# HANDLERS
# =========================
def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and int(user_id) == int(OWNER_ID)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ IMPULS запущен.\n\n"
        f"Канал: {CHANNEL_NAME}\n"
        f"Таймзона: {TIMEZONE_NAME}\n"
        f"Торговля: ПН–ПТ {TRADE_START}–{TRADE_END}\n"
        f"TF: {TF}\n"
        f"Expiry: {EXPIRY_MINUTES} мин\n\n"
        "Команды (владелец):\n"
        "/test\n/stats\n/report_now\n/pulse_on\n/pulse_off\n/diag\n/whoami\n",
        disable_web_page_preview=True,
    )

async def whoami_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id if update.effective_user else None
    await update.message.reply_text(f"👤 Your user_id: `{uid}`\nOWNER_ID in env: `{OWNER_ID}`", parse_mode=ParseMode.MARKDOWN)

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

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return

    ensure_day_reset()
    now = now_tz()
    header = (
        f"🧪 DIAG (V3.1)\n"
        f"Now: {now.strftime('%d.%m %H:%M:%S')} ({TIMEZONE_NAME})\n"
        f"TradingTime: {_bool_emoji(is_trading_time(now))}\n"
        f"TF={TF}  TrendFilter={int(TREND_FILTER)}({TREND_TF})\n"
        f"MIN_PROB={MIN_PROBABILITY}  ATR_BASE={ATR_THRESHOLD}  ADAPT={int(ADAPTIVE_FILTERS)}  ATR_MULT={GLOBAL_ATR_MULT}\n"
        f"TOP_N={TOP_N}  COOLDOWN={COOLDOWN_MINUTES}m\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        "—"
    )
    await update.message.reply_text(f"```text\n{header}\n```", parse_mode=ParseMode.MARKDOWN)

    lines: List[str] = []
    for s in SYMBOLS:
        try:
            d = diag_symbol(s)
            status = "PASS ✅" if d["pass"] else "REJECT ❌"
            cd = "CD ✅" if d["cooldown"] else "CD —"
            cand = d["candidate"] or "—"
            reason = "; ".join(d["reasons"]) if d["reasons"] else "ok"
            line = (
                f"{s:10} | {status:9} | {cd:4} | cand={cand:4} | "
                f"prob={d['prob']:>2}% | ATR={d['atrp']:.3f}% thr={d['thr']:.3f}% | "
                f"RSI={d['rsi']:.1f} | EMAtrend={d['local_trend']} | {reason}"
            )
            lines.append(line)
        except Exception as e:
            lines.append(f"{s:10} | ERROR ❌ | {e}")

    # отправляем кусками (чтобы Telegram не резал)
    chunk = []
    size = 0
    for ln in lines:
        if size + len(ln) + 1 > 3500:
            await update.message.reply_text("```text\n" + "\n".join(chunk) + "\n```", parse_mode=ParseMode.MARKDOWN)
            chunk, size = [], 0
        chunk.append(ln)
        size += len(ln) + 1
    if chunk:
        await update.message.reply_text("```text\n" + "\n".join(chunk) + "\n```", parse_mode=ParseMode.MARKDOWN)

    await update.message.reply_text(
        "Если везде REJECT из-за `ATR low` — снизь ATR_THRESHOLD или GLOBAL_ATR_MULT.\n"
        "Если из-за `RSI out` — расширим окна RSI.\n"
        "Скинь сюда результат /diag — я скажу точные правки под твой рынок."
    )

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()

    if not is_owner(q.from_user.id):
        await q.answer("⛔ Только владелец может отмечать WIN/LOSS.", show_alert=True)
        return

    data = (q.data or "").split("|")
    if len(data) != 3 or data[0] != "wl":
        return

    ensure_day_reset()
    action = data[1]
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
    app.add_handler(CommandHandler("diag", diag_cmd))
    app.add_handler(CallbackQueryHandler(on_button))

    if app.job_queue is None:
        raise RuntimeError("JobQueue не активен. Убедись, что установлен python-telegram-bot[job-queue]==22.5")

    app.job_queue.run_repeating(job_signals, interval=SIGNAL_INTERVAL_SECONDS, first=10, name="signals")
    app.job_queue.run_repeating(job_pulse, interval=PULSE_INTERVAL_SECONDS, first=60, name="pulse")

    report_t = time(hour=REPORT_HOUR, minute=REPORT_MINUTE, tzinfo=TZ)
    app.job_queue.run_daily(job_daily_report, time=report_t, name="daily_report")

    log.info(
        "IMPULS v3.1 started | TZ=%s | Trade=%s-%s | Symbols=%s | Mode=%s TOP_N=%s | TF=%s TrendTF=%s Trend=%s | ATR=%s adaptive=%s | expiry=%sm",
        TIMEZONE_NAME, TRADE_START, TRADE_END, SYMBOLS, SEND_MODE, TOP_N, TF, TREND_TF, TREND_FILTER,
        ATR_THRESHOLD, ADAPTIVE_FILTERS, EXPIRY_MINUTES
    )

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
