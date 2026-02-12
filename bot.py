# bot.py
# IMPULS ⚡ FINAL v4 — TwelveData, Hybrid Smart v2 (3m/5m), TOP-N, anti-limit (early-break + lazy trend),
# no-spam weak market, trading schedule (Mon–Fri 10:00–20:00), auto expiry report (no ID in text)
# Requires: python-telegram-bot[job-queue]==22.5

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
_raw_symbols = os.getenv("SYMBOLS", "EUR/USD,GBP/USD,USD/JPY,USD/CHF,EUR/JPY").strip()
# На всякий: если кто-то вставил "EUR/USD | GBP/USD" — превращаем в список
_raw_symbols = _raw_symbols.replace("|", ",").replace(";", ",")
SYMBOLS = [x.strip() for x in _raw_symbols.split(",") if x.strip()]

SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "600"))  # 10 минут
TF = os.getenv("TF", "1min").strip()
CANDLES = int(os.getenv("CANDLES", "250"))

# Гибрид (умный выбор экспирации)
HYBRID_MODE = os.getenv("HYBRID_MODE", "1").strip() in ("1", "true", "True", "YES", "yes")
SHORT_EXPIRY_MINUTES = int(os.getenv("SHORT_EXPIRY_MINUTES", "3"))   # быстрый импульс
LONG_EXPIRY_MINUTES = int(os.getenv("LONG_EXPIRY_MINUTES", "5"))     # спокойный тренд
# Порог импульса: выше → 3m, ниже → 5m
HYBRID_IMPULSE_CUTOFF = float(os.getenv("HYBRID_IMPULSE_CUTOFF", "0.72"))

# Фильтры
MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", "52"))
ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.006"))  # ATR% порог (в процентах)
ADAPTIVE_FILTERS = os.getenv("ADAPTIVE_FILTERS", "1").strip() in ("1", "true", "True", "YES", "yes")
GLOBAL_ATR_MULT = float(os.getenv("GLOBAL_ATR_MULT", "0.95"))

# Старший тренд (экономно проверяем только топ-кандидатов)
TREND_FILTER = os.getenv("TREND_FILTER", "1").strip() in ("1", "true", "True", "YES", "yes")
TREND_TF = os.getenv("TREND_TF", "15min").strip()
TREND_CHECK_TOP_K = int(os.getenv("TREND_CHECK_TOP_K", "2"))  # проверяем старший тренд только для ТОП-2 кандидатов

# Отправка сигналов
SEND_MODE = os.getenv("SEND_MODE", "TOP").strip().upper()  # TOP / BEST / ALL
TOP_N = int(os.getenv("TOP_N", "1"))

# Кулдаун на одну пару (чтобы не долбить один символ)
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "7"))

# Анти-спам сообщений “рынок слабый / не торговое время / лимит API”
WEAK_MSG_COOLDOWN_MINUTES = int(os.getenv("WEAK_MSG_COOLDOWN_MINUTES", "45"))
OFFTIME_MSG_COOLDOWN_MINUTES = int(os.getenv("OFFTIME_MSG_COOLDOWN_MINUTES", "60"))
APILIMIT_MSG_COOLDOWN_MINUTES = int(os.getenv("APILIMIT_MSG_COOLDOWN_MINUTES", "60"))

# Пульс (сообщение “бот жив”)
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "1800"))  # 30 минут
PULSE_ENABLED = os.getenv("PULSE_ENABLED", "1").strip() in ("1", "true", "True", "YES", "yes")

# Ежедневный отчёт (по умолчанию 20:05 в таймзоне)
REPORT_HOUR = int(os.getenv("REPORT_HOUR", "20"))
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", "5"))

# Умный early-break (чтобы меньше ловить лимиты)
EARLY_BREAK = os.getenv("EARLY_BREAK", "1").strip() in ("1", "true", "True", "YES", "yes")
EARLY_BREAK_PROB = int(os.getenv("EARLY_BREAK_PROB", "86"))  # если нашли очень сильный → стоп скан


# =========================
# ЛОГИ
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("impuls")


# =========================
# СТАТИСТИКА (в памяти, сброс по дню)
# =========================
STATS: Dict[str, Any] = {
    "day": None,  # YYYY-MM-DD (TZ)
    "signals": 0,
    "win": 0,
    "loss": 0,
    "last_signal": None,
    "pulse_on": True,
    "cooldown": {},       # symbol -> iso timestamp
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
    # ПН=0 ... ВС=6
    wd = dt.weekday()
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
        log.warning("OWNER_ID не задан (0). Owner-команды и WIN/LOSS недоступны.")


# =========================
# TWELVEDATA
# =========================
TD_BASE = "https://api.twelvedata.com"

def td_time_series(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    if not symbol:
        raise RuntimeError("symbol missing/invalid")

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
    if not symbol:
        raise RuntimeError("symbol missing/invalid")

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
        # порог = максимум из base и 0.80*медианы
        thr = max(base, 0.80 * med) * GLOBAL_ATR_MULT
        return float(thr)
    except Exception:
        return max(0.0, base) * GLOBAL_ATR_MULT


# =========================
# ЛОГИКА СИГНАЛА + ГИБРИД
# =========================
@dataclass
class Signal:
    symbol: str
    direction: str          # CALL/PUT
    probability: int
    price: float
    rsi14: float
    ema50: float
    ema200: float
    atr14_pct: float
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    expiry_minutes: int     # 3 или 5
    impulse: float          # 0..1 (примерно)

def compute_impulse(atrp: float, thr: float, ema50_v: float, ema200_v: float, price: float, rsi_v: float, direction: str) -> float:
    """
    Импульс 0..1:
    - ATR относительно порога (вола)
    - развод EMA в % (сила тренда)
    - RSI близость к “идеалу” для направления
    """
    if price <= 0:
        return 0.0

    # 1) Вола: 1.0 если atrp >= 2*thr, иначе пропорция
    vol = min(1.0, atrp / max(2.0 * thr, 0.0001))

    # 2) Развод EMA: 1.0 если >= 0.20% (для м1 это уже норм)
    spread_pct = abs(ema50_v - ema200_v) / price * 100.0
    spread = min(1.0, spread_pct / 0.20)

    # 3) RSI: близость к “рабочей” зоне
    # CALL: идеал около 55 (рабочая 45..65)
    # PUT : идеал около 45 (рабочая 35..55)
    ideal = 55.0 if direction.upper() == "CALL" else 45.0
    dist = abs(rsi_v - ideal)
    r = max(0.0, 1.0 - (dist / 20.0))  # 0..1

    # итог
    impulse = 0.45 * vol + 0.35 * spread + 0.20 * r
    return float(max(0.0, min(1.0, impulse)))

def choose_expiry(impulse: float) -> int:
    if not HYBRID_MODE:
        return SHORT_EXPIRY_MINUTES  # если гибрид выкл — фикс 3м
    return SHORT_EXPIRY_MINUTES if impulse >= HYBRID_IMPULSE_CUTOFF else LONG_EXPIRY_MINUTES

def trend_direction(symbol: str) -> Optional[str]:
    """
    Старший тренд (TREND_TF) — CALL если EMA50 > EMA200, PUT если EMA50 < EMA200.
    """
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

def compute_signal(symbol: str, df: pd.DataFrame) -> Optional[Signal]:
    """
    Считаем сигнал по уже загруженному df (экономия запросов).
    """
    df = df.copy()
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
        # CALL: RSI 45..65
        if 45 <= rsi_v <= 65:
            score += 35
            direction = "CALL"
    elif trend_down:
        score += 35
        # PUT: RSI 35..55
        if 35 <= rsi_v <= 55:
            score += 35
            direction = "PUT"
    else:
        return None

    if direction is None:
        return None

    # бонус за волу (относительно порога)
    rel = atrp / max(thr, 0.0001)
    vol_bonus = min(20, int(rel * 6))
    score += vol_bonus

    probability = max(55, min(92, int(score)))
    if probability < MIN_PROBABILITY:
        return None

    # импульс и экспирация
    impulse = compute_impulse(atrp, thr, ema50_v, ema200_v, close, rsi_v, direction)
    expiry = choose_expiry(impulse)

    entry = now_tz()
    exit_ = entry + timedelta(minutes=expiry)

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
        expiry_minutes=expiry,
        impulse=impulse,
    )

def in_cooldown(symbol: str) -> bool:
    ts = STATS["cooldown"].get(symbol)
    if not ts:
        return False
    ago = minutes_ago(ts)
    return (ago is not None) and (ago < COOLDOWN_MINUTES)

def mark_cooldown(symbol: str) -> None:
    STATS["cooldown"][symbol] = now_tz().isoformat()


# =========================
# TELEGRAM: сообщения
# =========================
def winloss_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ WIN", callback_data="wl|win"),
            InlineKeyboardButton("❌ LOSS", callback_data="wl|loss"),
        ]
    ])

def signal_message(sig: Signal) -> str:
    # короткий стиль “Pocket Option”
    exp = sig.expiry_minutes
    return (
        f"📊 *СИГНАЛ {sig.symbol}*\n"
        f"🎯 Направление: *{direction_label(sig.direction)}*\n"
        f"🔥 Вероятность: *{sig.probability}%*\n"
        f"⏳ Экспирация: *{exp} мин*\n\n"
        f"💰 Цена: `{sig.price:.5f}`\n"
        f"⚡ ATR(14): `{sig.atr14_pct:.3f}%`\n"
        f"📉 RSI(14): `{sig.rsi14:.1f}`\n"
        f"📍 EMA50/200: `{sig.ema50:.5f}` / `{sig.ema200:.5f}`\n\n"
        f"⏱ Вход: *{fmt_time(sig.entry_time)}*\n"
        f"🏁 Выход: *{fmt_time(sig.exit_time)}*\n"
        f"🌍 `{TIMEZONE_NAME}`"
    )

def offtime_message() -> str:
    return (
        f"🌙 Сейчас не торговое время.\n"
        f"📅 ПН–ПТ\n"
        f"⏰ {TRADE_START}–{TRADE_END} ({TIMEZONE_NAME})"
    )

def weak_market_message() -> str:
    return "📉 Рынок слабый — сильных сигналов нет. Продолжаю анализ…"

def api_limit_message() -> str:
    return (
        "⚠️ Лимит API/данные временно недоступны.\n"
        "Я восстановлюсь автоматически.\n"
        "Чтобы реже ловить лимит — увеличь интервал или сократи пары."
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
    expiry = int(data["expiry_minutes"])

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
        f"⏱ Экспирация *{expiry} мин* по *{symbol}*\n"
        f"📈 График пошёл: *{move_label}*\n"
        f"💰 Цена: `{entry_price:.5f}` → `{last_price:.5f}`  ({delta_str})\n"
        f"✅ По котировкам: *{quote_result}*\n\n"
        f"👉 Если у Pocket Option итог отличается — отметь вручную WIN/LOSS под сигналом."
    )
    await post_to_channel(context, text)


# =========================
# СКАНЕР (экономный) + TOP-N
# =========================
def pick_signals_efficient(symbols: List[str]) -> List[Signal]:
    """
    Экономия лимитов:
    1) сначала считаем сигналы по 1min для всех (по одному запросу на пару)
    2) сортируем кандидатов
    3) тренд-фильтр (если включен) проверяем только для ТОП-K кандидатов
    4) early-break: если нашли очень сильный — можно остановиться пораньше
    """
    candidates: List[Tuple[str, Signal]] = []

    # Кэш df1 для текущего цикла
    df_cache: Dict[str, pd.DataFrame] = {}

    for s in symbols:
        if in_cooldown(s):
            continue

        # 1 запрос на пару (1min)
        df1 = td_time_series(s, TF, CANDLES)
        df_cache[s] = df1

        sig = compute_signal(s, df1)
        if sig:
            candidates.append((s, sig))

            if EARLY_BREAK and sig.probability >= EARLY_BREAK_PROB and SEND_MODE in ("BEST", "TOP") and TOP_N == 1:
                # нашли очень сильный сигнал, экономим лимит
                break

    if not candidates:
        return []

    # сортируем по вероятности (и чуть-чуть по импульсу)
    candidates.sort(key=lambda x: (x[1].probability, x[1].impulse), reverse=True)

    # ALL — отдаём все
    if SEND_MODE == "ALL":
        return [c[1] for c in candidates]

    # BEST — только 1
    if SEND_MODE == "BEST":
        best = candidates[0][1]
        return [best]

    # TOP — берём top_n (но максимум 3 как ранее)
    n = max(1, min(3, TOP_N))
    top_list = [c[1] for c in candidates[:n]]

    # ТРЕНД фильтр: проверяем только для top-k кандидатов
    if TREND_FILTER:
        checked: List[Signal] = []
        to_check = top_list[:max(1, TREND_CHECK_TOP_K)]
        rest = top_list[max(1, TREND_CHECK_TOP_K):]

        for sig in to_check:
            tdir = trend_direction(sig.symbol)
            if tdir is None:
                checked.append(sig)  # нейтрально
            elif tdir == sig.direction:
                # бонус за совпадение тренда
                sig.probability = min(92, sig.probability + 5)
                checked.append(sig)
            else:
                # не совпало — выкидываем
                continue

        # добавляем остаток без проверки (чтобы не тратить лимит)
        checked.extend(rest)

        # пересортируем после бонусов/фильтра
        checked.sort(key=lambda x: (x.probability, x.impulse), reverse=True)

        # снова режем до n
        top_list = checked[:n]

    return top_list


# =========================
# JOBS
# =========================
async def job_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_day_reset()
    now = now_tz()

    # торговое время
    if not is_trading_time(now):
        if can_send_throttled("last_oftime_msg", OFFTIME_MSG_COOLDOWN_MINUTES):
            await post_to_channel(context, offtime_message())
        return

    # сигналим
    try:
        signals = pick_signals_efficient(SYMBOLS)
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

    # отправляем 1–3 лучших
    for sig in signals:
        STATS["signals"] += 1
        mark_cooldown(sig.symbol)
        STATS["last_signal"] = {
            "symbol": sig.symbol,
            "time": sig.entry_time.isoformat(),
            "prob": sig.probability,
            "exp": sig.expiry_minutes,
        }

        await post_to_channel(context, signal_message(sig), reply_markup=winloss_keyboard())

        # авто-отчёт после экспирации (exp*60 + 2 сек)
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
            name=f"expiry_{sig.symbol}_{sig.entry_time.strftime('%H%M%S')}",
        )

async def job_pulse(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not PULSE_ENABLED:
        return
    if not STATS.get("pulse_on", True):
        return
    # пульс только в торговое время
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
        f"📌 *{CHANNEL_NAME} — ОТЧЁТ ЗА ДЕНЬ*\n"
        f"🗓 Дата: *{now_tz().strftime('%d.%m.%Y')}*  ({TIMEZONE_NAME})\n\n"
        f"✉️ Сигналов: *{s}*\n"
        f"✅ WIN: *{w}*\n"
        f"❌ LOSS: *{l}*\n"
        f"🎯 WinRate: *{wr:.1f}%*"
    )
    await post_to_channel(context, txt)


# =========================
# HANDLERS
# =========================
def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ IMPULS запущен.\n\n"
        f"Канал: {CHANNEL_NAME}\n"
        f"Таймзона: {TIMEZONE_NAME}\n"
        f"Торговля: ПН–ПТ {TRADE_START}–{TRADE_END}\n"
        f"Гибрид: {'ON' if HYBRID_MODE else 'OFF'} (3m/5m)\n\n"
        "Команды (владелец):\n"
        "/test\n/stats\n/report_now\n/pulse_on\n/pulse_off\n",
        disable_web_page_preview=True,
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
    await update.message.reply_text("✅ Пульс выключен.")

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()

    if not is_owner(q.from_user.id):
        await q.answer("⛔ Только владелец может отмечать WIN/LOSS.", show_alert=True)
        return

    data = (q.data or "").split("|")
    if len(data) != 2 or data[0] != "wl":
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
    app.add_handler(CommandHandler("test", test_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("report_now", report_now_cmd))
    app.add_handler(CommandHandler("pulse_on", pulse_on_cmd))
    app.add_handler(CommandHandler("pulse_off", pulse_off_cmd))
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
        "IMPULS v4 started | TZ=%s | Trade=%s-%s | Symbols=%s | Mode=%s TOP_N=%s | TF=%s Trend=%s(%s) | "
        "ATR=%.4f adaptive=%s mult=%.2f | Hybrid=%s 3m=%s 5m=%s cutoff=%.2f | interval=%ss",
        TIMEZONE_NAME, TRADE_START, TRADE_END, SYMBOLS, SEND_MODE, TOP_N, TF, TREND_FILTER, TREND_TF,
        ATR_THRESHOLD, ADAPTIVE_FILTERS, GLOBAL_ATR_MULT,
        HYBRID_MODE, SHORT_EXPIRY_MINUTES, LONG_EXPIRY_MINUTES, HYBRID_IMPULSE_CUTOFF,
        SIGNAL_INTERVAL_SECONDS
    )

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
