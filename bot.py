# bot_FINAL_v4.py
# IMPULS ⚡ FINAL v4 — TwelveData, TOP-N, hybrid expiry (3m/5m), early-break, anti-spam, trading schedule,
# async-safe (no event-loop blocking), auto expiry report, owner-only controls
# python-telegram-bot[job-queue]==22.5

import os
import logging
import asyncio
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
OWNER_ID = int(os.getenv("OWNER_ID", "0") or "0")
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "IMPULS ⚡").strip()

TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Kyiv").strip()
TZ = ZoneInfo(TIMEZONE_NAME)

# Торговые дни/время (ПН–ПТ)
TRADE_START = os.getenv("TRADE_START", "10:00").strip()  # HH:MM
TRADE_END = os.getenv("TRADE_END", "20:00").strip()      # HH:MM

# Сканер
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "720"))  # 12 минут по умолчанию
TF = os.getenv("TF", "1min").strip()
CANDLES = int(os.getenv("CANDLES", "250"))

SEND_MODE = os.getenv("SEND_MODE", "TOP").strip().upper()  # TOP / BEST / ALL
TOP_N = int(os.getenv("TOP_N", "1"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "7"))

# Фильтры качества
MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", "52"))

ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.006"))   # ATR% базовый порог
ADAPTIVE_FILTERS = os.getenv("ADAPTIVE_FILTERS", "1").strip() in ("1", "true", "True", "YES", "yes")
GLOBAL_ATR_MULT = float(os.getenv("GLOBAL_ATR_MULT", "0.95"))

TREND_FILTER = os.getenv("TREND_FILTER", "0").strip() in ("1", "true", "True", "YES", "yes")
TREND_TF = os.getenv("TREND_TF", "15min").strip()

# HYBRID expiry (3m/5m)
HYBRID_MODE = os.getenv("HYBRID_MODE", "1").strip() in ("1", "true", "True", "YES", "yes")
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "3"))  # базово (если гибрид выключен)
EXPIRY_FAST = int(os.getenv("EXPIRY_FAST", "3"))        # короткая экспирация
EXPIRY_SLOW = int(os.getenv("EXPIRY_SLOW", "5"))        # длинная экспирация
HYBRID_ATR_BORDER = float(os.getenv("HYBRID_ATR_BORDER", "0.014"))  # ATR% граница: выше = быстрее
HYBRID_CHOP_THRESHOLD = float(os.getenv("HYBRID_CHOP_THRESHOLD", "61"))  # чем выше, тем более "пила" -> длиннее
HYBRID_IMPULSE_CUTOFF = float(os.getenv("HYBRID_IMPULSE_CUTOFF", "1.15"))  # импульс: выше -> быстрее
HYBRID_MIN_PROB = int(os.getenv("HYBRID_MIN_PROB", str(MIN_PROBABILITY)))

# EARLY BREAK (экономит лимит API): если нашли достаточно сильный сигнал — не сканим остальные пары
EARLY_BREAK = os.getenv("EARLY_BREAK", "1").strip() in ("1", "true", "True", "YES", "yes")
EARLY_BREAK_PROB = int(os.getenv("EARLY_BREAK_PROB", "78"))
TREND_CHECK_TOP_K = int(os.getenv("TREND_CHECK_TOP_K", "0"))  # 0 = не применять (оставлено для совместимости)

# Анти-спам сообщений
WEAK_MSG_COOLDOWN_MINUTES = int(os.getenv("WEAK_MSG_COOLDOWN_MINUTES", "45"))
OFFTIME_MSG_COOLDOWN_MINUTES = int(os.getenv("OFFTIME_MSG_COOLDOWN_MINUTES", "60"))
APILIMIT_MSG_COOLDOWN_MINUTES = int(os.getenv("APILIMIT_MSG_COOLDOWN_MINUTES", "60"))

# Пульс
PULSE_ENABLED = os.getenv("PULSE_ENABLED", "1").strip() in ("1", "true", "True", "YES", "yes")
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "1800"))  # 30 минут

# Ежедневный отчёт (по умолчанию 20:05)
REPORT_HOUR = int(os.getenv("REPORT_HOUR", "20"))
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", "5"))

# Пары
def _parse_symbols(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        raw = "EUR/USD,GBP/USD,USD/JPY,USD/CHF,EUR/JPY"
    # частая ошибка: вставляют "SYMBOLS=EUR/USD,..." прямо в value
    raw = raw.replace("SYMBOLS=", "").replace("symbols=", "")
    # иногда разделяют через | или ;
    raw = raw.replace("|", ",").replace(";", ",")
    out = []
    for x in raw.split(","):
        s = x.strip()
        if not s:
            continue
        out.append(s)
    return out

SYMBOLS = _parse_symbols(os.getenv("SYMBOLS", ""))


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
    "day": None,
    "signals": 0,
    "win": 0,
    "loss": 0,
    "last_signal": None,
    "pulse_on": True,
    "cooldown": {},  # symbol -> iso ts
    "last_weak_msg": None,
    "last_oftime_msg": None,
    "last_api_msg": None,
}


# =========================
# ВРЕМЯ
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
    wd = dt.weekday()  # 0..6
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
        raise RuntimeError("BOT_TOKEN пустой. Добавь BOT_TOKEN в Variables.")
    if not TWELVE_API_KEY:
        raise RuntimeError("TWELVE_API_KEY пустой. Добавь TWELVE_API_KEY в Variables.")
    if not CHANNEL_ID:
        raise RuntimeError("CHANNEL_ID пустой. Добавь CHANNEL_ID в Variables.")
    if OWNER_ID == 0:
        log.warning("OWNER_ID не задан (0). Owner-команды будут недоступны.")


# =========================
# TWELVEDATA (SYNC) — будет вызываться через asyncio.to_thread
# =========================
TD_BASE = "https://api.twelvedata.com"

def td_time_series_sync(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
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

def td_price_sync(symbol: str) -> float:
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

async def td_time_series(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    return await asyncio.to_thread(td_time_series_sync, symbol, interval, outputsize)

async def td_price(symbol: str) -> float:
    return await asyncio.to_thread(td_price_sync, symbol)


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

def choppiness_index(df: pd.DataFrame, period: int = 14) -> float:
    """
    Примерный Choppiness (0..100): чем выше — тем больше "пила".
    Формула по ATR sum / range.
    """
    try:
        if len(df) < period + 2:
            return 50.0
        a = atr_series(df, 1).tail(period).sum()
        hi = df["high"].tail(period).max()
        lo = df["low"].tail(period).min()
        rng = float(hi - lo)
        if rng <= 0:
            return 50.0
        # 100*log10(sumATR/range)/log10(period) (классика), но нам достаточно масштаба
        import math
        return float(100.0 * math.log10(max(a, 1e-9) / rng) / math.log10(period))
    except Exception:
        return 50.0


# =========================
# СИГНАЛ
# =========================
@dataclass
class Signal:
    symbol: str
    direction: str  # CALL/PUT
    probability: int
    price: float
    atr14_pct: float
    rsi14: float
    ema50: float
    ema200: float
    entry_time: datetime
    exit_time: datetime
    expiry_minutes: int
    entry_price: float

def direction_arrow(direction: str) -> str:
    return "⬆️ ВВЕРХ" if direction.upper() == "CALL" else "⬇️ ВНИЗ"

async def trend_ok(symbol: str) -> Optional[str]:
    """
    Старший тренд (TREND_TF):
    CALL если EMA50 > EMA200, PUT если EMA50 < EMA200.
    """
    if not TREND_FILTER:
        return None
    df = await td_time_series(symbol, TREND_TF, 220)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    e50 = float(df["ema50"].iloc[-1])
    e200 = float(df["ema200"].iloc[-1])
    if e50 > e200:
        return "CALL"
    if e50 < e200:
        return "PUT"
    return None

def pick_expiry_minutes(atrp: float, chop: float, impulse: float) -> int:
    if not HYBRID_MODE:
        return EXPIRY_MINUTES
    # если рынок "пилит" — лучше дольше
    if chop >= HYBRID_CHOP_THRESHOLD:
        return EXPIRY_SLOW
    # сильный импульс и вола — быстрее
    if (atrp >= HYBRID_ATR_BORDER) and (impulse >= HYBRID_IMPULSE_CUTOFF):
        return EXPIRY_FAST
    # иначе — дольше (стабильнее)
    return EXPIRY_SLOW

async def compute_signal(symbol: str) -> Optional[Signal]:
    df = await td_time_series(symbol, TF, CANDLES)

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

    # локальный тренд
    trend_up = ema50_v > ema200_v
    trend_down = ema50_v < ema200_v
    if not trend_up and not trend_down:
        return None

    direction = None
    score = 0

    if trend_up:
        score += 35
        if 43 <= rsi_v <= 67:
            score += 35
            direction = "CALL"
    elif trend_down:
        score += 35
        if 31 <= rsi_v <= 59:
            score += 35
            direction = "PUT"

    if direction is None:
        return None

    # старший тренд: совпадает — бонус, не совпадает — бан
    if TREND_FILTER:
        tdir = await trend_ok(symbol)
        if tdir and tdir != direction:
            return None
        if tdir and tdir == direction:
            score += 12

    rel = atrp / max(thr, 0.0001)
    vol_bonus = min(20, int(rel * 6))
    score += vol_bonus

    probability = max(50, min(92, int(score)))
    min_prob = HYBRID_MIN_PROB if HYBRID_MODE else MIN_PROBABILITY
    if probability < min_prob:
        return None

    # гибрид: выбираем экспирацию (3/5)
    chop = choppiness_index(df, 14)
    impulse = rel  # простая метрика импульса: atrp/thr
    expiry = pick_expiry_minutes(atrp, chop, impulse)

    entry = now_tz()
    exit_ = entry + timedelta(minutes=expiry)

    return Signal(
        symbol=symbol,
        direction=direction,
        probability=probability,
        price=close,
        atr14_pct=atrp,
        rsi14=rsi_v,
        ema50=ema50_v,
        ema200=ema200_v,
        entry_time=entry,
        exit_time=exit_,
        expiry_minutes=expiry,
        entry_price=close,
    )


# =========================
# COOLDOWN
# =========================
def in_cooldown(symbol: str) -> bool:
    ts = STATS["cooldown"].get(symbol)
    if not ts:
        return False
    ago = minutes_ago(ts)
    return (ago is not None) and (ago < COOLDOWN_MINUTES)

def mark_cooldown(symbol: str) -> None:
    STATS["cooldown"][symbol] = now_tz().isoformat()


# =========================
# TELEGRAM: UI
# =========================
def winloss_keyboard(signal_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ WIN", callback_data=f"wl|win|{signal_id}"),
            InlineKeyboardButton("❌ LOSS", callback_data=f"wl|loss|{signal_id}"),
        ]
    ])

def signal_message(sig: Signal) -> str:
    # формат как ты просил (HTML + эмодзи)
    return (
        f"📊 <b>СИГНАЛ {sig.symbol}</b>\n"
        f"🎯 <b>Направление:</b> {direction_arrow(sig.direction)}\n"
        f"🔥 <b>Вероятность:</b> {sig.probability}%\n"
        f"⌛️ <b>Экспирация:</b> {sig.expiry_minutes} мин\n\n"
        f"⏱ <b>Вход:</b> {fmt_time(sig.entry_time)}\n"
        f"🏁 <b>Выход:</b> {fmt_time(sig.exit_time)}\n"
        f"🌍 <b>{TIMEZONE_NAME}</b>"
    )

def expiry_message(symbol: str, expiry_minutes: int, direction: str, entry_price: float, last_price: float) -> str:
    # результат “по котировкам”
    quote_win = (last_price > entry_price) if direction.upper() == "CALL" else (last_price < entry_price)
    result = "✅ WIN" if quote_win else "❌ LOSS"
    move = "⬆️ ВВЕРХ" if last_price > entry_price else ("⬇️ ВНИЗ" if last_price < entry_price else "➡️ РОВНО")
    return (
        f"⏱ <b>Экспирация {expiry_minutes} мин</b> по <b>{symbol}</b>\n"
        f"📈 <b>График пошёл:</b> {move}\n"
        f"✅ <b>По котировкам:</b> {result}"
    )

def offtime_message() -> str:
    return (
        f"🌙 <b>Сейчас не торговое время.</b>\n"
        f"📅 Торгую ПН–ПТ\n"
        f"⏰ {TRADE_START}–{TRADE_END} (<b>{TIMEZONE_NAME}</b>)"
    )

def weak_market_message() -> str:
    return "📉 <b>Рынок слабый — сильных сигналов нет.</b>\nЯ продолжаю анализировать…"

def api_limit_message() -> str:
    return (
        "⚠️ <b>Данные временно недоступны (лимит API).</b>\n"
        "Я не молчу — провайдер ограничил запросы.\n"
        "Уменьши частоту/список пар или подожди."
    )

async def post_to_channel(context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None) -> None:
    await context.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )


# =========================
# JOBS
# =========================
async def job_expiry_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data or {}
    symbol = data["symbol"]
    direction = data["direction"]
    entry_price = float(data["entry_price"])
    expiry_minutes = int(data.get("expiry_minutes", EXPIRY_MINUTES))

    try:
        last_price = await td_price(symbol)
    except Exception as e:
        log.warning("Expiry price fetch failed for %s: %s", symbol, e)
        return

    await post_to_channel(context, expiry_message(symbol, expiry_minutes, direction, entry_price, last_price))

async def job_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_day_reset()
    now = now_tz()

    if not is_trading_time(now):
        if can_send_throttled("last_oftime_msg", OFFTIME_MSG_COOLDOWN_MINUTES):
            await post_to_channel(context, offtime_message())
        return

    found: List[Signal] = []
    try:
        for s in SYMBOLS:
            if in_cooldown(s):
                continue

            sig = await compute_signal(s)
            if sig:
                found.append(sig)

                # EARLY_BREAK: если нашли уже очень сильный — не тратим кредиты
                if EARLY_BREAK and sig.probability >= EARLY_BREAK_PROB:
                    break

        found.sort(key=lambda x: x.probability, reverse=True)

    except Exception as e:
        msg = str(e).lower()
        if ("credit" in msg) or ("limit" in msg) or ("429" in msg) or ("too many" in msg):
            if can_send_throttled("last_api_msg", APILIMIT_MSG_COOLDOWN_MINUTES):
                await post_to_channel(context, api_limit_message())
            return
        log.exception("Signal scan error: %s", e)
        return

    if not found:
        if can_send_throttled("last_weak_msg", WEAK_MSG_COOLDOWN_MINUTES):
            await post_to_channel(context, weak_market_message())
        return

    # режим выдачи
    if SEND_MODE == "ALL":
        to_send = found
    elif SEND_MODE == "BEST":
        to_send = found[:1]
    else:
        to_send = found[:max(1, TOP_N)]

    for sig in to_send:
        STATS["signals"] += 1
        mark_cooldown(sig.symbol)

        signal_id = f"{sig.entry_time.strftime('%Y%m%d%H%M%S')}_{sig.symbol.replace('/', '')}"
        STATS["last_signal"] = {"symbol": sig.symbol, "time": sig.entry_time.isoformat(), "prob": sig.probability}

        await post_to_channel(context, signal_message(sig), reply_markup=winloss_keyboard(signal_id))

        delay = sig.expiry_minutes * 60 + 2
        context.job_queue.run_once(
            job_expiry_report,
            when=delay,
            data={
                "symbol": sig.symbol,
                "direction": sig.direction,
                "entry_price": sig.entry_price,
                "expiry_minutes": sig.expiry_minutes,
            },
            name=f"expiry_{signal_id}",
        )

async def job_pulse(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not PULSE_ENABLED or not STATS.get("pulse_on", True):
        return
    if not is_trading_time(now_tz()):
        return
    await post_to_channel(context, f"🕒 <b>{CHANNEL_NAME}</b>: бот жив, анализирую рынок…")

async def job_daily_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_day_reset()
    s = STATS["signals"]
    w = STATS["win"]
    l = STATS["loss"]
    wr = (w / max(1, w + l)) * 100.0

    txt = (
        f"📌 <b>{CHANNEL_NAME} — ЕЖЕДНЕВНЫЙ ОТЧЁТ</b>\n"
        f"🗓 <b>Дата:</b> {now_tz().strftime('%d.%m.%Y')} (<b>{TIMEZONE_NAME}</b>)\n\n"
        f"✉️ <b>Сигналов:</b> {s}\n"
        f"✅ <b>WIN:</b> {w}\n"
        f"❌ <b>LOSS:</b> {l}\n"
        f"🎯 <b>WinRate:</b> {wr:.1f}%"
    )
    await post_to_channel(context, txt)


# =========================
# HANDLERS
# =========================
def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Ответ всегда — даже если OWNER_ID не совпал
    hybrid_txt = "ON (3m/5m)" if HYBRID_MODE else f"OFF ({EXPIRY_MINUTES}m)"
    await update.message.reply_text(
        f"✅ <b>IMPULS запущен.</b>\n\n"
        f"Канал: <b>{CHANNEL_NAME}</b>\n"
        f"Таймзона: <b>{TIMEZONE_NAME}</b>\n"
        f"Торговля: <b>ПН–ПТ {TRADE_START}–{TRADE_END}</b>\n"
        f"Гибрид: <b>{hybrid_txt}</b>\n"
        f"TF: <b>{TF}</b>\n"
        f"Symbols: <b>{', '.join(SYMBOLS)}</b>\n\n"
        f"<b>Команды (владелец):</b>\n"
        f"/test\n/stats\n/report_now\n/pulse_on\n/pulse_off\n/whoami\n/debug_pairs\n",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

async def whoami_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id if update.effective_user else 0
    await update.message.reply_text(
        f"🆔 user_id: <b>{uid}</b>\n"
        f"👑 owner: <b>{'YES' if is_owner(uid) else 'NO'}</b>\n"
        f"OWNER_ID env: <b>{OWNER_ID}</b>",
        parse_mode=ParseMode.HTML,
    )

async def test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    await post_to_channel(context, "✅ <b>ТЕСТ:</b> бот может писать в канал (OK)")
    await update.message.reply_text("✅ Тест отправлен в канал.", parse_mode=ParseMode.HTML)

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
        f"📊 <b>Статистика (за сегодня)</b>\n"
        f"Сигналов: <b>{s}</b>\n"
        f"WIN: <b>{w}</b>\n"
        f"LOSS: <b>{l}</b>\n"
        f"WinRate: <b>{wr:.1f}%</b>\n"
        f"Последний: <code>{last if last else '—'}</code>",
        parse_mode=ParseMode.HTML,
    )

async def report_now_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    await job_daily_report(context)
    await update.message.reply_text("✅ Отчёт отправлен в канал.", parse_mode=ParseMode.HTML)

async def pulse_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    STATS["pulse_on"] = True
    await update.message.reply_text("✅ Пульс включён.", parse_mode=ParseMode.HTML)

async def pulse_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    STATS["pulse_on"] = False
    await update.message.reply_text("✅ Пульс выключен.", parse_mode=ParseMode.HTML)

async def debug_pairs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Диагностика: что проходит/не проходит по парам (может тратить кредиты TwelveData).
    """
    if not is_owner(update.effective_user.id):
        return

    lines = []
    try:
        for s in SYMBOLS:
            if in_cooldown(s):
                lines.append(f"{s}: CD ⏳")
                continue
            try:
                sig = await compute_signal(s)
                if sig:
                    lines.append(f"{s}: PASS ✅ | prob={sig.probability}% | exp={sig.expiry_minutes}m | ATR={sig.atr14_pct:.3f}% | RSI={sig.rsi14:.1f}")
                else:
                    lines.append(f"{s}: REJECT ❌")
            except Exception as e:
                lines.append(f"{s}: ERROR ❌ | {str(e)[:80]}")
    except Exception as e:
        await update.message.reply_text(f"ERROR: {e}", parse_mode=ParseMode.HTML)
        return

    await update.message.reply_text("<b>DEBUG PAIRS</b>\n" + "\n".join(lines), parse_mode=ParseMode.HTML)

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
        await q.message.reply_text("✅ WIN отмечен", parse_mode=ParseMode.HTML)
    elif action == "loss":
        STATS["loss"] += 1
        await q.message.reply_text("❌ LOSS отмечен", parse_mode=ParseMode.HTML)


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
        raise RuntimeError("JobQueue не активен. Убедись, что установлен python-telegram-bot[job-queue]==22.5")

    # Сигналы
    app.job_queue.run_repeating(job_signals, interval=SIGNAL_INTERVAL_SECONDS, first=10, name="signals")

    # Пульс
    app.job_queue.run_repeating(job_pulse, interval=PULSE_INTERVAL_SECONDS, first=60, name="pulse")

    # Ежедневный отчёт (по TZ)
    report_t = time(hour=REPORT_HOUR, minute=REPORT_MINUTE, tzinfo=TZ)
    app.job_queue.run_daily(job_daily_report, time=report_t, name="daily_report")

    log.info(
        "IMPULS started | TZ=%s | Trade=%s-%s | Symbols=%s | Mode=%s TOP_N=%s | TF=%s | Trend=%s TrendTF=%s | Hybrid=%s (fast=%sm slow=%sm) | Interval=%ss",
        TIMEZONE_NAME, TRADE_START, TRADE_END, SYMBOLS, SEND_MODE, TOP_N, TF,
        TREND_FILTER, TREND_TF, HYBRID_MODE, EXPIRY_FAST, EXPIRY_SLOW, SIGNAL_INTERVAL_SECONDS
    )

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
