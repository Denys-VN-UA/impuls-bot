# bot.py
# IMPULS ⚡ — версия FINNHUB (OANDA forex candles) + таймзона + стрелки направления
# python-telegram-bot[job-queue]==22.5

import os
import math
import logging
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Optional, Any, List

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# =========================
# НАСТРОЙКИ (ENV)
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

# Источник данных
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "FINNHUB").strip().upper()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

# Канал куда бот постит сигналы:
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()  # обязателен на Railway

# Владелец (только он может /stats /report_now /pulse_on /pulse_off и ставить WIN/LOSS)
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

# Название канала (в тексте)
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "IMPULS ⚡")

# Таймзона
TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Kyiv")
TZ = ZoneInfo(TIMEZONE_NAME)

# Расписание / интервалы
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "180"))  # 3 минуты
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "600"))    # 10 минут
REPORT_HOUR = int(os.getenv("REPORT_HOUR", "22"))
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", "0"))

# Пары для сканера
DEFAULT_SYMBOLS = os.getenv(
    "SYMBOLS",
    "USD/JPY,USD/CHF,EUR/USD,GBP/USD,EUR/JPY,GBP/JPY,AUD/USD,USD/CAD"
).split(",")

# Порог волатильности (ATR%) — можно уменьшать если мало сигналов
ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.010"))

# Таймфрейм и количество свечей
TF = os.getenv("TF", "1min")        # 1min/5min/15min/30min/60min
CANDLES = int(os.getenv("CANDLES", "250"))

# Экспирация (мин)
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "3"))

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
STATS = {
    "signals": 0,
    "win": 0,
    "loss": 0,
    "last_signal": None,  # dict
    "pulse_on": True,
}

# =========================
# УТИЛИТЫ
# =========================
def now_tz() -> datetime:
    return datetime.now(TZ)

def fmt_dt(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%d.%m.%Y %H:%M:%S")

def fmt_time(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%H:%M:%S")

def direction_label(direction: str) -> str:
    # direction internally: 'CALL' or 'PUT'
    if direction.upper() == "CALL":
        return "⬆️ ВВЕРХ"
    return "⬇️ ВНИЗ"

def require_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN пустой. Добавь переменную окружения BOT_TOKEN.")
    if not CHANNEL_ID:
        raise RuntimeError("CHANNEL_ID пустой. Добавь переменную окружения CHANNEL_ID.")
    if DATA_PROVIDER == "FINNHUB" and not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY пустой. Добавь переменную окружения FINNHUB_API_KEY.")
    if OWNER_ID == 0:
        log.warning("OWNER_ID не задан (0). Команды owner-only и WIN/LOSS будут недоступны.")

# =========================
# FINNHUB DATA (Forex candles via OANDA symbols)
# =========================
FH_BASE = "https://finnhub.io/api/v1"

def tf_to_resolution(tf: str):
    tf = (tf or "").strip().lower()
    mapping = {
        "1min": 1,
        "3min": 3,
        "5min": 5,
        "10min": 10,
        "15min": 15,
        "30min": 30,
        "60min": 60,
        "1h": 60,
        "d": "D",
        "1d": "D",
    }
    return mapping.get(tf, 1)

def to_finnhub_forex_symbol(pair: str) -> str:
    """
    'EUR/USD' -> 'OANDA:EUR_USD'
    Если уже задано как 'OANDA:...' — возвращаем как есть.
    """
    p = pair.strip().upper()
    if ":" in p:
        return p
    if "/" not in p:
        return p
    base, quote = p.split("/", 1)
    return f"OANDA:{base}_{quote}"

def finnhub_time_series_forex(pair: str, interval: str, outputsize: int = 250) -> pd.DataFrame:
    """
    Возвращает OHLC dataframe по Finnhub (forex/candle).
    """
    symbol = to_finnhub_forex_symbol(pair)
    resolution = tf_to_resolution(interval)

    # Период: берём с запасом чуть больше, чем outputsize
    now_utc = datetime.utcnow()
    if resolution == "D":
        seconds_back = (outputsize + 50) * 86400
    else:
        seconds_back = (outputsize + 50) * int(resolution) * 60

    _to = int(now_utc.timestamp())
    _from = _to - seconds_back

    url = f"{FH_BASE}/forex/candle"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": _from,
        "to": _to,
        "token": FINNHUB_API_KEY,
    }

    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    if data.get("s") != "ok":
        raise RuntimeError(f"Finnhub error for {pair} ({symbol}): {data}")

    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["t"], unit="s", utc=True),
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
    })

    # оставляем последние outputsize свечей
    df = df.tail(outputsize).reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df

def get_time_series(symbol: str, interval: str, outputsize: int = 250) -> pd.DataFrame:
    if DATA_PROVIDER == "FINNHUB":
        return finnhub_time_series_forex(symbol, interval, outputsize)
    raise RuntimeError(f"Unknown DATA_PROVIDER={DATA_PROVIDER}")

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
# ЛОГИКА СИГНАЛА
# =========================
@dataclass
class Signal:
    symbol: str
    direction: str  # CALL/PUT internal
    probability: int
    price: float
    rsi14: float
    ema50: float
    ema200: float
    atr14_pct: float
    entry_time: datetime
    exit_time: datetime
    reason: str

def compute_signal(symbol: str) -> Optional[Signal]:
    df = get_time_series(symbol, TF, CANDLES)

    if len(df) < 210:
        return None

    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)
    atr_pct = atr_percent(df, 14)

    if atr_pct < ATR_THRESHOLD:
        return None

    close = float(df["close"].iloc[-1])
    ema50_v = float(df["ema50"].iloc[-1])
    ema200_v = float(df["ema200"].iloc[-1])
    rsi_v = float(df["rsi14"].iloc[-1])

    trend_up = ema50_v > ema200_v
    trend_down = ema50_v < ema200_v

    direction = None
    score = 0
    reasons = []

    if trend_up:
        score += 35
        reasons.append("EMA50 выше EMA200 (тренд вверх)")
        if 45 <= rsi_v <= 65:
            score += 35
            reasons.append("RSI в зоне импульса вверх")
            direction = "CALL"
        else:
            reasons.append("RSI не подтверждает вверх")
    elif trend_down:
        score += 35
        reasons.append("EMA50 ниже EMA200 (тренд вниз)")
        if 35 <= rsi_v <= 55:
            score += 35
            reasons.append("RSI в зоне импульса вниз")
            direction = "PUT"
        else:
            reasons.append("RSI не подтверждает вниз")
    else:
        return None

    vol_bonus = min(20, int((atr_pct / max(ATR_THRESHOLD, 0.0001)) * 5))
    score += vol_bonus
    reasons.append(f"ATR(14) {atr_pct:.3f}%")

    if direction is None:
        return None

    probability = max(55, min(92, int(score)))
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
        atr14_pct=atr_pct,
        entry_time=entry,
        exit_time=exit_,
        reason=" | ".join(reasons),
    )

def pick_best_signal(symbols: List[str]) -> Optional[Signal]:
    best = None
    for s in symbols:
        s = s.strip()
        if not s:
            continue
        try:
            sig = compute_signal(s)
        except Exception as e:
            log.warning("Signal error for %s: %s", s, e)
            continue

        if not sig:
            continue

        if best is None or sig.probability > best.probability:
            best = sig
    return best

# =========================
# TELEGRAM: сообщения
# =========================
def signal_message(sig: Signal) -> str:
    dir_text = direction_label(sig.direction)
    return (
        f"📊 *СИГНАЛ {sig.symbol}*\n"
        f"📈 Направление: *{dir_text}*\n"
        f"🔥 Вероятность: *{sig.probability}%*\n\n"
        f"💰 Цена: `{sig.price:.5f}`\n"
        f"📉 RSI(14): `{sig.rsi14:.1f}`\n"
        f"📍 EMA50: `{sig.ema50:.5f}`\n"
        f"📍 EMA200: `{sig.ema200:.5f}`\n"
        f"⚡ ATR(14): `{sig.atr14_pct:.3f}%`\n\n"
        f"⏱ Вход: *{fmt_time(sig.entry_time)}*\n"
        f"🏁 Выход: *{fmt_time(sig.exit_time)}*  (экспирация {EXPIRY_MINUTES} мин)\n"
        f"🌍 Таймзона: `{TIMEZONE_NAME}`\n"
        f"🛰 Data: `{DATA_PROVIDER}`\n"
    )

def winloss_keyboard(signal_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ WIN", callback_data=f"wl|win|{signal_id}"),
            InlineKeyboardButton("❌ LOSS", callback_data=f"wl|loss|{signal_id}"),
        ]
    ])

async def post_to_channel(context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None) -> None:
    await context.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )

# =========================
# JOBS
# =========================
async def job_send_best_signal(context: ContextTypes.DEFAULT_TYPE) -> None:
    sig = pick_best_signal(DEFAULT_SYMBOLS)
    if not sig:
        return

    STATS["signals"] += 1
    signal_id = sig.entry_time.strftime("%Y%m%d%H%M%S")
    STATS["last_signal"] = {"id": signal_id, "symbol": sig.symbol, "ts": fmt_dt(sig.entry_time)}

    msg = signal_message(sig)
    await post_to_channel(context, msg, reply_markup=winloss_keyboard(signal_id))

async def job_pulse(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not STATS.get("pulse_on", True):
        return
    await post_to_channel(context, f"🕒 *{CHANNEL_NAME}*: бот жив, анализирую рынок…")

async def job_daily_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    s = STATS["signals"]
    w = STATS["win"]
    l = STATS["loss"]
    wr = (w / max(1, w + l)) * 100.0

    txt = (
        f"📌 *{CHANNEL_NAME} — ЕЖЕДНЕВНЫЙ ОТЧЁТ*\n"
        f"🗓 Дата: *{now_tz().strftime('%d.%m.%Y')}*  ({TIMEZONE_NAME})\n\n"
        f"📨 Сигналов: *{s}*\n"
        f"✅ WIN: *{w}*\n"
        f"❌ LOSS: *{l}*\n"
        f"🎯 WinRate: *{wr:.1f}%*\n"
        f"🛰 Data: `{DATA_PROVIDER}`\n"
    )
    await post_to_channel(context, txt)

# =========================
# HANDLERS
# =========================
def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ Бот активен.\n"
        f"Канал: {CHANNEL_NAME}\n"
        f"Таймзона: {TIMEZONE_NAME}\n"
        f"Источник: {DATA_PROVIDER}\n\n"
        "Команды (только владелец):\n"
        "/test — тест в канал\n"
        "/stats — статистика\n"
        "/report_now — отчёт сейчас\n"
        "/pulse_on — включить пульс\n"
        "/pulse_off — выключить пульс\n",
        disable_web_page_preview=True,
    )

async def test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    await post_to_channel(context, "✅ *ТЕСТ*: бот может писать в канал (OK)")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    s = STATS["signals"]
    w = STATS["win"]
    l = STATS["loss"]
    wr = (w / max(1, w + l)) * 100.0
    last = STATS.get("last_signal")
    last_txt = f"{last}" if last else "—"

    await update.message.reply_text(
        f"📊 Статистика\n"
        f"Сигналов: {s}\n"
        f"WIN: {w}\n"
        f"LOSS: {l}\n"
        f"WinRate: {wr:.1f}%\n"
        f"Последний: {last_txt}\n"
        f"Таймзона: {TIMEZONE_NAME}\n"
        f"Источник: {DATA_PROVIDER}",
    )

async def report_now(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    await job_daily_report(context)
    await update.message.reply_text("✅ Отчёт отправлен в канал.")

async def pulse_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    STATS["pulse_on"] = True
    await update.message.reply_text("✅ Пульс включён (раз в 10 минут сообщение в канал).")

async def pulse_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    STATS["pulse_on"] = False
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

    if action == "win":
        STATS["win"] += 1
        await q.message.reply_text(f"✅ Отмечено: WIN (signal_id={signal_id})")
    elif action == "loss":
        STATS["loss"] += 1
        await q.message.reply_text(f"❌ Отмечено: LOSS (signal_id={signal_id})")

# =========================
# MAIN
# =========================
def main() -> None:
    require_env()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("test", test_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("report_now", report_now))
    app.add_handler(CommandHandler("pulse_on", pulse_on))
    app.add_handler(CommandHandler("pulse_off", pulse_off))
    app.add_handler(CallbackQueryHandler(on_button))

    if app.job_queue is None:
        raise RuntimeError("JobQueue не активен. Убедись, что установлен пакет python-telegram-bot[job-queue]==22.5")

    app.job_queue.run_repeating(job_send_best_signal, interval=SIGNAL_INTERVAL_SECONDS, first=10)
    app.job_queue.run_repeating(job_pulse, interval=PULSE_INTERVAL_SECONDS, first=60)

    report_t = time(hour=REPORT_HOUR, minute=REPORT_MINUTE, tzinfo=TZ)
    app.job_queue.run_daily(job_daily_report, time=report_t, name="daily_report")

    log.info("%s | бот запущен | TZ=%s | report=%02d:%02d | DATA=%s",
             CHANNEL_NAME, TIMEZONE_NAME, REPORT_HOUR, REPORT_MINUTE, DATA_PROVIDER)

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
