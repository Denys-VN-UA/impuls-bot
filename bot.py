# bot.py
# IMPULS ⚡ — финальная PRO-версия (больше WIN) + таймзона + стрелки + отчёт
# Требования: python-telegram-bot[job-queue]==22.5, requests, pandas, numpy

import os
import logging
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Optional, Any, List

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "").strip()

# Канал куда бот постит сигналы:
# Вариант 1: numeric id -100xxxxxxxxxx
# Вариант 2: @username_channel
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()

# Владелец (только он может /stats /report_now /pulse_on /pulse_off и ставить WIN/LOSS)
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

# Название канала (в тексте)
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "IMPULS ⚡")

# Таймзона
TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Kyiv")
TZ = ZoneInfo(TIMEZONE_NAME)

# Интервалы
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "180"))  # 3 минуты
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "600"))    # 10 минут

# Ежедневный отчёт (по таймзоне)
REPORT_HOUR = int(os.getenv("REPORT_HOUR", "22"))
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", "0"))

# Пары/символы TwelveData (можешь расширить)
DEFAULT_SYMBOLS = os.getenv(
    "SYMBOLS",
    "USD/JPY,USD/CHF,EUR/USD,GBP/USD,EUR/JPY,GBP/JPY,AUD/USD,USD/CAD,BTC/USD,ETH/USD"
).split(",")

# Таймфрейм и количество свечей
TF = os.getenv("TF", "1min")
CANDLES = int(os.getenv("CANDLES", "260"))

# Экспирация Pocket Option
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "3"))

# ===== PRO фильтры (больше WIN, меньше сигналов) =====
ADX_PERIOD = 14
ADX_MIN = int(os.getenv("ADX_MIN", "22"))
ADX_MAX = int(os.getenv("ADX_MAX", "40"))

# ATR в % (для Forex + Crypto вместе)
ATR_MIN = float(os.getenv("ATR_MIN", "0.025"))
ATR_MAX = float(os.getenv("ATR_MAX", "0.12"))

# EMA gap (в %)
EMA_GAP_MIN = float(os.getenv("EMA_GAP_MIN", "0.03"))
EMA_GAP_MAX = float(os.getenv("EMA_GAP_MAX", "0.25"))

# RSI зоны под 3 минуты (Pocket Option)
RSI_UP_MIN = float(os.getenv("RSI_UP_MIN", "48"))
RSI_UP_MAX = float(os.getenv("RSI_UP_MAX", "62"))
RSI_DN_MIN = float(os.getenv("RSI_DN_MIN", "38"))
RSI_DN_MAX = float(os.getenv("RSI_DN_MAX", "52"))

# Свечной фильтр (импульс)
BODY_RATIO_MIN = float(os.getenv("BODY_RATIO_MIN", "0.60"))

# Минимальная вероятность для отправки (держит частоту ~5–10/день)
MIN_PROB_TO_SEND = int(os.getenv("MIN_PROB_TO_SEND", "78"))

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
    """Internal CALL/PUT -> RU arrows"""
    if direction.upper() == "CALL":
        return "⬆️ ВВЕРХ"
    return "⬇️ ВНИЗ"

def require_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN пустой. Добавь переменную окружения BOT_TOKEN.")
    if not TWELVE_API_KEY:
        raise RuntimeError("TWELVE_API_KEY пустой. Добавь переменную окружения TWELVE_API_KEY.")
    if not CHANNEL_ID:
        raise RuntimeError("CHANNEL_ID пустой. Добавь переменную окружения CHANNEL_ID.")
    if OWNER_ID == 0:
        log.warning("OWNER_ID не задан (0). Команды owner-only и WIN/LOSS будут недоступны.")

def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

# =========================
# TWELVE DATA
# =========================
TD_BASE = "https://api.twelvedata.com"

def td_time_series(symbol: str, interval: str, outputsize: int = 200) -> pd.DataFrame:
    """OHLC dataframe from TwelveData"""
    url = f"{TD_BASE}/time_series"
    params = {
        "symbol": symbol.strip(),
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_API_KEY,
        "format": "JSON",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    if data.get("status") == "error":
        raise RuntimeError(f"TwelveData error for {symbol}: {data.get('message')}")

    values = data.get("values") or []
    if not values:
        raise RuntimeError(f"No candles returned for {symbol}")

    df = pd.DataFrame(values)
    # Latest->oldest -> reverse
    df = df.iloc[::-1].reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df

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

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    atr_ = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1/period, adjust=False).mean()

def candle_body_ratio(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    rng = float(last["high"] - last["low"])
    if rng <= 0:
        return 0.0
    body = abs(float(last["close"] - last["open"]))
    return float(body / rng)

# =========================
# СИГНАЛ
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
    adx14: float
    atr14_pct: float
    entry_time: datetime
    exit_time: datetime
    reason: str

def compute_signal(symbol: str) -> Optional[Signal]:
    df = td_time_series(symbol, TF, CANDLES)

    # индикаторы
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)
    df["adx14"] = adx(df, ADX_PERIOD)

    atr_pct = atr_percent(df, 14)
    if atr_pct < ATR_MIN or atr_pct > ATR_MAX:
        return None

    close = float(df["close"].iloc[-1])
    ema50_v = float(df["ema50"].iloc[-1])
    ema200_v = float(df["ema200"].iloc[-1])
    rsi_v = float(df["rsi14"].iloc[-1])
    adx_v = float(df["adx14"].iloc[-1])

    # 1) ADX фильтр
    if adx_v < ADX_MIN or adx_v > ADX_MAX:
        return None

    # 2) EMA gap фильтр (анти-флэт + анти-перегрев)
    ema_gap_pct = abs(ema50_v - ema200_v) / close * 100.0 if close else 0.0
    if ema_gap_pct < EMA_GAP_MIN or ema_gap_pct > EMA_GAP_MAX:
        return None

    # 3) свечной фильтр
    br = candle_body_ratio(df)
    if br < BODY_RATIO_MIN:
        return None

    # 4) направление + RSI зоны
    trend_up = ema50_v > ema200_v
    trend_down = ema50_v < ema200_v

    direction = None
    if trend_up:
        if close <= ema50_v:
            return None
        if not (RSI_UP_MIN <= rsi_v <= RSI_UP_MAX):
            return None
        direction = "CALL"
    elif trend_down:
        if close >= ema50_v:
            return None
        if not (RSI_DN_MIN <= rsi_v <= RSI_DN_MAX):
            return None
        direction = "PUT"
    else:
        return None

    # Probability (строго, для 5–10 сигналов/день)
    prob = 72

    # ADX bonus
    if 22 <= adx_v <= 28:
        prob += 6
    elif 28 < adx_v <= 35:
        prob += 9
    else:
        prob += 4

    # ATR bonus (комфортная вола)
    if 0.03 <= atr_pct <= 0.09:
        prob += 6
    else:
        prob += 3

    # EMA gap bonus (чистый тренд)
    if 0.05 <= ema_gap_pct <= 0.18:
        prob += 5
    else:
        prob += 2

    # candle bonus
    if br >= 0.70:
        prob += 4
    else:
        prob += 2

    probability = int(max(70, min(92, prob)))

    entry = now_tz()
    exit_ = entry + timedelta(minutes=EXPIRY_MINUTES)

    return Signal(
        symbol=symbol.strip(),
        direction=direction,
        probability=probability,
        price=close,
        rsi14=rsi_v,
        ema50=ema50_v,
        ema200=ema200_v,
        adx14=adx_v,
        atr14_pct=atr_pct,
        entry_time=entry,
        exit_time=exit_,
        reason=f"ADX={adx_v:.1f} | ATR%={atr_pct:.3f} | GAP%={ema_gap_pct:.3f} | BODY={br:.2f}",
    )

def pick_best_signal(symbols: List[str]) -> Optional[Signal]:
    best: Optional[Signal] = None
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
        f"📈 ADX(14): `{sig.adx14:.1f}`\n"
        f"⚡ ATR(14): `{sig.atr14_pct:.3f}%`\n\n"
        f"⏱ Вход: *{fmt_time(sig.entry_time)}*\n"
        f"🏁 Выход: *{fmt_time(sig.exit_time)}*  (экспирация {EXPIRY_MINUTES} мин)\n"
        f"🌍 Таймзона: `{TIMEZONE_NAME}`\n"
        f"🧠 PRO: `{sig.reason}`"
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

    # Частота/качество: не отправляем ниже порога
    if sig.probability < MIN_PROB_TO_SEND:
        return

    STATS["signals"] += 1
    signal_id = sig.entry_time.strftime("%Y%m%d%H%M%S")
    STATS["last_signal"] = {"id": signal_id, "symbol": sig.symbol, "ts": fmt_dt(sig.entry_time)}

    msg = signal_message(sig)
    await post_to_channel(context, msg, reply_markup=winloss_keyboard(signal_id))
    log.info("SENT %s %s prob=%s", sig.symbol, sig.direction, sig.probability)

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
        f"⚙️ MIN_PROB_TO_SEND: *{MIN_PROB_TO_SEND}%*"
    )
    await post_to_channel(context, txt)

# =========================
# HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ Бот активен (Pocket Option: только сигналы).\n"
        f"Канал: {CHANNEL_NAME}\n"
        f"Таймзона: {TIMEZONE_NAME}\n"
        f"Экспирация: {EXPIRY_MINUTES} мин\n"
        f"Порог отправки: {MIN_PROB_TO_SEND}%\n\n"
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
        f"Таймзона: {TIMEZONE_NAME}",
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
    await update.message.reply_text("✅ Пульс включён.")

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
        raise RuntimeError("JobQueue не активен. Установи python-telegram-bot[job-queue]==22.5")

    # Сканер сигналов
    app.job_queue.run_repeating(job_send_best_signal, interval=SIGNAL_INTERVAL_SECONDS, first=10)

    # Пульс
    app.job_queue.run_repeating(job_pulse, interval=PULSE_INTERVAL_SECONDS, first=60)

    # Ежедневный отчёт
    report_t = time(hour=REPORT_HOUR, minute=REPORT_MINUTE, tzinfo=TZ)
    app.job_queue.run_daily(job_daily_report, time=report_t, name="daily_report")

    log.info("%s | бот запущен | TZ=%s | report=%02d:%02d | MIN_PROB=%d",
             CHANNEL_NAME, TIMEZONE_NAME, REPORT_HOUR, REPORT_MINUTE, MIN_PROB_TO_SEND)

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
