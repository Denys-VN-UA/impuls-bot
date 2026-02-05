# bot.py
# IMPULS ⚡ — финальная версия (TwelveData)
# ✅ TOP_N (1–3 лучших) вместо 6 подряд
# ✅ Никогда не молчит (но анти-спам)
# ✅ ПН–ПТ 10:00–20:00 (Europe/Kyiv), СБ/ВС выходной
# ✅ Адаптивный ATR (ADAPTIVE_FILTERS=1)
# ✅ Фильтр старшего тренда (TREND_FILTER=1, TREND_TF=15min/60min)
# ✅ Авто-отчёт после экспирации + авто-определение куда пошёл график (WIN/LOSS)
# python-telegram-bot[job-queue]==22.5

import os
import logging
import math
import time as time_mod
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, Tuple, List

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# =========================
# ENV / НАСТРОЙКИ
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "").strip()
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()  # -100xxxx or @channel
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Kyiv").strip()
TZ = ZoneInfo(TIMEZONE_NAME)

# Список пар
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "EUR/USD,USD/JPY,USD/CHF").split(",") if s.strip()]

# Интервалы
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "600"))  # лучше 600 для лимитов API
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "600"))
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "3"))
ENTRY_DELAY_SECONDS = int(os.getenv("ENTRY_DELAY_SECONDS", "0"))  # если хочешь вход с задержкой

# Режим отправки
# BEST = отправить 1 лучший
# TOP  = отправить TOP_N лучших (1..3)
# ALL  = отправить все, что прошло фильтры
SEND_MODE = os.getenv("SEND_MODE", "TOP").strip().upper()
TOP_N = int(os.getenv("TOP_N", "2"))
MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", "60"))  # ниже — не отправляем
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "6"))  # чтобы не спамить одной парой

# TF для входа
TF = os.getenv("TF", "1min").strip()
CANDLES = int(os.getenv("CANDLES", "250"))

# ATR фильтр (в %)
ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.020"))  # 0.020% — мягко
ADAPTIVE_FILTERS = os.getenv("ADAPTIVE_FILTERS", "0").strip()  # 1=вкл
GLOBAL_ATR_MULT = float(os.getenv("GLOBAL_ATR_MULT", "1.00"))   # множитель к медиане ATR (если adaptive)

# Фильтр старшего тренда
TREND_FILTER = os.getenv("TREND_FILTER", "1").strip()  # 1=вкл
TREND_TF = os.getenv("TREND_TF", "15min").strip()      # 15min или 60min
TREND_CANDLES = int(os.getenv("TREND_CANDLES", "300"))

# Торговое время (ПН–ПТ 10:00–20:00)
TRADE_START = os.getenv("TRADE_START", "10:00").strip()
TRADE_END = os.getenv("TRADE_END", "20:00").strip()

# Ежедневный отчёт
REPORT_HOUR = int(os.getenv("REPORT_HOUR", "20"))
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", "0"))

CHANNEL_NAME = os.getenv("CHANNEL_NAME", "IMPULS ⚡").strip()

# =========================
# ЛОГИ
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("impuls")

# =========================
# STATE / СТАТИСТИКА
# =========================
STATE = {
    "pulse_on": True,
    "last_no_signal_ts": 0.0,
    "last_api_limit_ts": 0.0,
    "cooldown_until": {},  # symbol -> datetime
    "pending": {},         # signal_id -> dict(signal)
}

STATS = {
    "day": None,   # 'YYYY-MM-DD'
    "signals": 0,
    "win": 0,
    "loss": 0,
}

# =========================
# HELPERS
# =========================
def now_tz() -> datetime:
    return datetime.now(TZ)

def fmt_time(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%H:%M:%S")

def today_key() -> str:
    return now_tz().strftime("%Y-%m-%d")

def reset_daily_if_needed() -> None:
    d = today_key()
    if STATS["day"] != d:
        STATS["day"] = d
        STATS["signals"] = 0
        STATS["win"] = 0
        STATS["loss"] = 0

def parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm), tzinfo=TZ)

TRADE_START_T = parse_hhmm(TRADE_START)
TRADE_END_T = parse_hhmm(TRADE_END)

def is_trading_time(dt: datetime) -> bool:
    # СБ/ВС выходной
    if dt.weekday() >= 5:
        return False
    t = dt.timetz()
    return (t >= TRADE_START_T) and (t < TRADE_END_T)

def direction_label(direction: str) -> str:
    return "⬆️ ВВЕРХ" if direction.upper() == "CALL" else "⬇️ ВНИЗ"

def direction_arrow(direction: str) -> str:
    return "📈" if direction.upper() == "CALL" else "📉"

def require_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN пустой. Добавь BOT_TOKEN в Railway Variables.")
    if not TWELVE_API_KEY:
        raise RuntimeError("TWELVE_API_KEY пустой. Добавь TWELVE_API_KEY в Railway Variables.")
    if not CHANNEL_ID:
        raise RuntimeError("CHANNEL_ID пустой. Добавь CHANNEL_ID в Railway Variables.")

# =========================
# TWELVEDATA
# =========================
TD_BASE = "https://api.twelvedata.com"

class ApiLimitError(RuntimeError):
    pass

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

    if data.get("status") == "error":
        msg = str(data.get("message", "unknown error"))
        # типичная ошибка лимита — пусть будет отдельной
        if "API credits" in msg or "run out of API credits" in msg or "limit" in msg.lower():
            raise ApiLimitError(msg)
        raise RuntimeError(f"TwelveData error for {symbol}: {msg}")

    values = data.get("values") or []
    if not values:
        raise RuntimeError(f"No candles for {symbol}")

    df = pd.DataFrame(values).iloc[::-1].reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df

def td_quote_price(symbol: str) -> float:
    url = f"{TD_BASE}/price"
    params = {"symbol": symbol, "apikey": TWELVE_API_KEY, "format": "JSON"}
    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    if data.get("status") == "error":
        msg = str(data.get("message", "unknown error"))
        if "API credits" in msg or "run out of API credits" in msg or "limit" in msg.lower():
            raise ApiLimitError(msg)
        raise RuntimeError(f"TwelveData price error for {symbol}: {msg}")

    p = data.get("price")
    return float(p)

# =========================
# CACHE (экономим лимит API)
# =========================
_TS_CACHE: Dict[Tuple[str, str, int], Tuple[float, pd.DataFrame]] = {}
_PRICE_CACHE: Dict[str, Tuple[float, float]] = {}  # symbol -> (ts, price)

def td_time_series_cached(symbol: str, interval: str, outputsize: int, ttl_sec: int) -> pd.DataFrame:
    key = (symbol, interval, outputsize)
    now = time_mod.time()
    hit = _TS_CACHE.get(key)
    if hit and (now - hit[0] < ttl_sec):
        return hit[1]
    df = td_time_series(symbol, interval, outputsize)
    _TS_CACHE[key] = (now, df)
    return df

def td_quote_cached(symbol: str, ttl_sec: int = 8) -> float:
    now = time_mod.time()
    hit = _PRICE_CACHE.get(symbol)
    if hit and (now - hit[0] < ttl_sec):
        return hit[1]
    p = td_quote_price(symbol)
    _PRICE_CACHE[symbol] = (now, p)
    return p

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

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
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
    a = atr(df, period).iloc[-1]
    c = df["close"].iloc[-1]
    if c == 0 or pd.isna(a) or pd.isna(c):
        return 0.0
    return float((a / c) * 100.0)

# =========================
# HIGHER TF TREND FILTER
# =========================
def higher_tf_trend(symbol: str) -> Optional[str]:
    # Старший ТФ меняется редко → кэш 10 минут
    df = td_time_series_cached(symbol, TREND_TF, TREND_CANDLES, ttl_sec=600)
    df["ema50_htf"] = ema(df["close"], 50)
    df["ema200_htf"] = ema(df["close"], 200)

    e50 = float(df["ema50_htf"].iloc[-1])
    e200 = float(df["ema200_htf"].iloc[-1])
    close = float(df["close"].iloc[-1])
    if close == 0:
        return None

    # “мертвая зона” (очень близко — лучше пропустить)
    diff_pct = abs(e50 - e200) / close * 100.0
    if diff_pct < 0.003:
        return None

    return "CALL" if e50 > e200 else "PUT"

# =========================
# SIGNAL LOGIC
# =========================
@dataclass
class Signal:
    symbol: str
    direction: str      # CALL / PUT
    probability: int
    price: float
    rsi14: float
    ema50: float
    ema200: float
    atr14_pct: float
    entry_time: datetime
    exit_time: datetime

def compute_signal(symbol: str, atr_thr: float) -> Optional[Signal]:
    df = td_time_series_cached(symbol, TF, CANDLES, ttl_sec=30)

    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)

    atr_pct = atr_percent(df, 14)
    if atr_pct < atr_thr:
        return None

    close = float(df["close"].iloc[-1])
    ema50_v = float(df["ema50"].iloc[-1])
    ema200_v = float(df["ema200"].iloc[-1])
    rsi_v = float(df["rsi14"].iloc[-1])

    trend_up = ema50_v > ema200_v
    trend_down = ema50_v < ema200_v

    direction = None
    score = 0

    if trend_up:
        score += 35
        # зона для M1 (мягкая)
        if 44 <= rsi_v <= 68:
            score += 35
            direction = "CALL"
    elif trend_down:
        score += 35
        if 32 <= rsi_v <= 56:
            score += 35
            direction = "PUT"
    else:
        return None

    if direction is None:
        return None

    # Фильтр старшего тренда
    if TREND_FILTER == "1":
        htf = higher_tf_trend(symbol)
        if htf is None:
            return None
        if direction != htf:
            return None
        score += 10

    # бонус за волатильность
    vol_bonus = min(20, int((atr_pct / max(atr_thr, 0.0001)) * 5))
    score += vol_bonus

    probability = max(55, min(92, int(score)))
    if probability < MIN_PROBABILITY:
        return None

    entry = now_tz() + timedelta(seconds=ENTRY_DELAY_SECONDS)
    exit_ = entry + timedelta(minutes=EXPIRY_MINUTES)

    return Signal(
        symbol=symbol,
        direction=direction,
        probability=probability,
        price=close,
        rsi14=rsi_v,
        ema50=ema50_v,
        ema200=ema200_v,
        atr14_pct=atr_pct,
        entry_time=entry,
        exit_time=exit_,
    )

def compute_adaptive_atr_threshold(symbols: List[str]) -> float:
    """Медиана ATR% по парам * GLOBAL_ATR_MULT (мягкая адаптация)."""
    atrs: List[float] = []
    for s in symbols:
        try:
            df = td_time_series_cached(s, TF, min(CANDLES, 200), ttl_sec=45)
            atrs.append(atr_percent(df, 14))
        except Exception:
            continue
    if not atrs:
        return ATR_THRESHOLD
    med = float(np.median(atrs))
    return max(0.005, min(0.150, med * GLOBAL_ATR_MULT))

# =========================
# TELEGRAM TEXT
# =========================
def signal_text(sig: Signal, signal_id: str) -> str:
    # Упрощённый стиль “Pocket Option”
    return (
        f"📊 *СИГНАЛ {sig.symbol}* {direction_arrow(sig.direction)}\n"
        f"🎯 Направление: *{direction_label(sig.direction)}*\n"
        f"🔥 Вероятность: *{sig.probability}%*\n\n"
        f"💰 Цена: `{sig.price:.5f}`\n"
        f"⚡ ATR(14): `{sig.atr14_pct:.3f}%`\n"
        f"📉 RSI(14): `{sig.rsi14:.1f}`\n"
        f"📍 EMA50: `{sig.ema50:.5f}`\n"
        f"📍 EMA200: `{sig.ema200:.5f}`\n\n"
        f"⏱ Вход: *{fmt_time(sig.entry_time)}*\n"
        f"🏁 Выход: *{fmt_time(sig.exit_time)}*  (эксп. {EXPIRY_MINUTES} мин)\n"
        f"🌍 Таймзона: `{TIMEZONE_NAME}`\n"
        f"🆔 id: `{signal_id}`"
    )

def winloss_keyboard(signal_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ WIN", callback_data=f"wl|win|{signal_id}"),
        InlineKeyboardButton("❌ LOSS", callback_data=f"wl|loss|{signal_id}"),
    ]])

async def post_channel(context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None) -> None:
    await context.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )

# =========================
# AUTO RESULT AFTER EXPIRY
# =========================
def outcome_from_prices(direction: str, entry_price: float, exit_price: float) -> str:
    if direction.upper() == "CALL":
        return "WIN" if exit_price > entry_price else "LOSS"
    return "WIN" if exit_price < entry_price else "LOSS"

async def job_after_expiry(context: ContextTypes.DEFAULT_TYPE) -> None:
    reset_daily_if_needed()

    signal_id = context.job.data.get("signal_id")
    if not signal_id:
        return
    rec = STATE["pending"].pop(signal_id, None)
    if not rec:
        return

    symbol = rec["symbol"]
    direction = rec["direction"]
    entry_price = rec["entry_price"]
    entry_ts = rec["entry_time"]

    # Подождём 2–3 секунды, чтобы “последняя свеча/цена” точно обновилась
    try:
        exit_price = td_quote_cached(symbol, ttl_sec=0)
    except ApiLimitError:
        # не спамим
        return
    except Exception:
        return

    delta = exit_price - entry_price
    delta_pct = 0.0 if entry_price == 0 else (delta / entry_price * 100.0)

    move = "⬆️" if delta > 0 else ("⬇️" if delta < 0 else "➡️")
    result = outcome_from_prices(direction, entry_price, exit_price)

    if result == "WIN":
        STATS["win"] += 1
    else:
        STATS["loss"] += 1

    await post_channel(
        context,
        (
            f"⏱ *Экспирация прошла* по *{symbol}*\n"
            f"📍 Было: `{entry_price:.5f}` → Стало: `{exit_price:.5f}`  ({move} `{delta_pct:+.3f}%`)\n"
            f"🎯 Сигнал: *{direction_label(direction)}*  → Итог: *{('✅ WIN' if result=='WIN' else '❌ LOSS')}*\n"
            f"🆔 id: `{signal_id}`"
        )
    )

# =========================
# DAILY REPORT
# =========================
async def job_daily_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    reset_daily_if_needed()
    s = STATS["signals"]
    w = STATS["win"]
    l = STATS["loss"]
    wr = (w / max(1, w + l)) * 100.0

    await post_channel(
        context,
        (
            f"📌 *{CHANNEL_NAME} — ЕЖЕДНЕВНЫЙ ОТЧЁТ*\n"
            f"🗓 Дата: *{now_tz().strftime('%d.%m.%Y')}*  (`{TIMEZONE_NAME}`)\n\n"
            f"📨 Сигналов: *{s}*\n"
            f"✅ WIN: *{w}*\n"
            f"❌ LOSS: *{l}*\n"
            f"🎯 WinRate: *{wr:.1f}%*"
        )
    )

# =========================
# JOBS
# =========================
async def job_pulse(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not STATE.get("pulse_on", True):
        return
    await post_channel(context, f"🕒 *{CHANNEL_NAME}*: бот жив, анализирую рынок…")

def in_cooldown(symbol: str, dt: datetime) -> bool:
    until = STATE["cooldown_until"].get(symbol)
    return bool(until and dt < until)

def set_cooldown(symbol: str, dt: datetime) -> None:
    STATE["cooldown_until"][symbol] = dt + timedelta(minutes=COOLDOWN_MINUTES)

async def job_send_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    reset_daily_if_needed()
    dt = now_tz()

    # Торговое расписание
    if not is_trading_time(dt):
        # анти-спам: не чаще раза в 60 минут
        if time_mod.time() - STATE["last_no_signal_ts"] > 3600:
            STATE["last_no_signal_ts"] = time_mod.time()
            await post_channel(
                context,
                f"🌙 *Сейчас не торговое время.*\n"
                f"📅 Торгую ПН–ПТ\n"
                f"⏰ {TRADE_START}–{TRADE_END} (`{TIMEZONE_NAME}`)"
            )
        return

    # ATR threshold (адаптивный)
    atr_thr = ATR_THRESHOLD
    if ADAPTIVE_FILTERS == "1":
        atr_thr = compute_adaptive_atr_threshold(SYMBOLS)

    signals: List[Signal] = []
    api_limited = False

    for sym in SYMBOLS:
        if in_cooldown(sym, dt):
            continue
        try:
            sig = compute_signal(sym, atr_thr)
        except ApiLimitError:
            api_limited = True
            break
        except Exception as e:
            log.warning("Signal error for %s: %s", sym, e)
            continue

        if sig:
            signals.append(sig)

    # Если лимит API — один раз в 15 минут
    if api_limited:
        if time_mod.time() - STATE["last_api_limit_ts"] > 900:
            STATE["last_api_limit_ts"] = time_mod.time()
            await post_channel(
                context,
                "⚠️ *Данные временно недоступны (лимит API).* \n"
                "Я не молчу — провайдер ограничил запросы. Попробуй позже или уменьши частоту/список пар."
            )
        return

    if not signals:
        # “никогда не молчит”, но без спама: раз в 20 минут
        if time_mod.time() - STATE["last_no_signal_ts"] > 1200:
            STATE["last_no_signal_ts"] = time_mod.time()
            await post_channel(
                context,
                f"📉 *Рынок слабый / нет сильных сигналов*.\n"
                f"Фильтры: ATR≥`{atr_thr:.3f}%`, minProb≥`{MIN_PROBABILITY}%`"
            )
        return

    # сортируем по вероятности
    signals.sort(key=lambda x: x.probability, reverse=True)

    if SEND_MODE == "BEST":
        send_list = signals[:1]
    elif SEND_MODE == "ALL":
        send_list = signals
    else:
        send_list = signals[:max(1, min(3, TOP_N))]  # 1..3

    for sig in send_list:
        # фиксируем entry/exit
        signal_id = f"{sig.entry_time.strftime('%Y%m%d%H%M%S')}_{sig.symbol.replace('/','')}"
        STATS["signals"] += 1

        # entry price (берем quote, чтобы точнее)
        try:
            entry_price = td_quote_cached(sig.symbol, ttl_sec=0)
        except Exception:
            entry_price = sig.price

        # сохраняем pending для авто-результата
        STATE["pending"][signal_id] = {
            "symbol": sig.symbol,
            "direction": sig.direction,
            "entry_price": float(entry_price),
            "entry_time": sig.entry_time,
            "exit_time": sig.exit_time,
        }

        # отправляем сигнал
        await post_channel(
            context,
            signal_text(sig, signal_id),
            reply_markup=winloss_keyboard(signal_id),
        )

        # ставим кулдаун по паре
        set_cooldown(sig.symbol, dt)

        # планируем авто-результат после экспирации (+2 сек)
        delay = max(5, int((sig.exit_time - now_tz()).total_seconds()) + 2)
        context.job_queue.run_once(
            job_after_expiry,
            when=delay,
            data={"signal_id": signal_id},
            name=f"after_{signal_id}",
        )

# =========================
# HANDLERS
# =========================
def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"✅ {CHANNEL_NAME} активен.\n"
        f"Таймзона: {TIMEZONE_NAME}\n"
        f"Пары: {', '.join(SYMBOLS)}\n"
        f"Режим: {SEND_MODE}, TOP_N={TOP_N}\n"
        f"Торгую: ПН–ПТ {TRADE_START}–{TRADE_END}\n\n"
        "Owner команды:\n"
        "/test\n"
        "/stats\n"
        "/report_now\n"
        "/pulse_on\n"
        "/pulse_off\n",
        disable_web_page_preview=True,
    )

async def test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    await post_channel(context, "✅ *ТЕСТ*: бот может писать в канал (OK)")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    reset_daily_if_needed()
    w, l, s = STATS["win"], STATS["loss"], STATS["signals"]
    wr = (w / max(1, w + l)) * 100.0
    await update.message.reply_text(
        f"📊 Статистика за сегодня ({STATS['day']})\n"
        f"Сигналов: {s}\n"
        f"WIN: {w}\n"
        f"LOSS: {l}\n"
        f"WinRate: {wr:.1f}%\n"
        f"Торгую: ПН–ПТ {TRADE_START}–{TRADE_END} ({TIMEZONE_NAME})"
    )

async def report_now_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    await job_daily_report(context)
    await update.message.reply_text("✅ Отчёт отправлен в канал.")

async def pulse_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    STATE["pulse_on"] = True
    await update.message.reply_text("✅ Пульс включён.")

async def pulse_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    STATE["pulse_on"] = False
    await update.message.reply_text("✅ Пульс выключён.")

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()

    user_id = q.from_user.id
    if not is_owner(user_id):
        await q.answer("⛔ Только владелец может отмечать WIN/LOSS.", show_alert=True)
        return

    data = (q.data or "").split("|")
    if len(data) != 3 or data[0] != "wl":
        return

    action = data[1]
    signal_id = data[2]

    reset_daily_if_needed()

    if action == "win":
        STATS["win"] += 1
        await q.message.reply_text(f"✅ WIN отмечен (id={signal_id})")
    elif action == "loss":
        STATS["loss"] += 1
        await q.message.reply_text(f"❌ LOSS отмечен (id={signal_id})")

# =========================
# MAIN
# =========================
def main() -> None:
    require_env()
    reset_daily_if_needed()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("test", test_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("report_now", report_now_cmd))
    app.add_handler(CommandHandler("pulse_on", pulse_on_cmd))
    app.add_handler(CommandHandler("pulse_off", pulse_off_cmd))
    app.add_handler(CallbackQueryHandler(on_button))

    if app.job_queue is None:
        raise RuntimeError("JobQueue не активен. Установи python-telegram-bot[job-queue]==22.5")

    # Сканер сигналов
    app.job_queue.run_repeating(job_send_signals, interval=SIGNAL_INTERVAL_SECONDS, first=10, name="signals")

    # Пульс
    app.job_queue.run_repeating(job_pulse, interval=PULSE_INTERVAL_SECONDS, first=60, name="pulse")

    # Ежедневный отчёт + гарантированный дневной reset “на границе”
    report_t = time(hour=REPORT_HOUR, minute=REPORT_MINUTE, tzinfo=TZ)
    app.job_queue.run_daily(job_daily_report, time=report_t, name="daily_report")

    log.info(
        "%s | старт | TZ=%s | trade=%s-%s | symbols=%d | mode=%s top=%d | interval=%ds",
        CHANNEL_NAME, TIMEZONE_NAME, TRADE_START, TRADE_END, len(SYMBOLS), SEND_MODE, TOP_N, SIGNAL_INTERVAL_SECONDS
    )

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
