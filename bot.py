# bot.py
# IMPULS ⚡ — TOP-1/2/3 лучших + авто-отчёт после каждой сделки + адаптивные фильтры
# Требования: python-telegram-bot[job-queue]==22.5

import os
import logging
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Tuple

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

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

SYMBOLS = [s.strip() for s in os.getenv(
    "SYMBOLS",
    "USD/JPY,USD/CHF,EUR/USD,GBP/USD,EUR/JPY,GBP/JPY,AUD/USD,USD/CAD"
).split(",") if s.strip()]

TF = os.getenv("TF", "1min").strip()
CANDLES = int(os.getenv("CANDLES", "250") or "250")
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "3") or "3")

SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "180") or "180")
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "600") or "600")
REPORT_HOUR = int(os.getenv("REPORT_HOUR", "22") or "22")
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", "0") or "0")

SEND_MODE = os.getenv("SEND_MODE", "TOP").strip().upper()  # TOP / ALL
TOP_N = int(os.getenv("TOP_N", "2") or "2")
TOP_N = max(1, min(3, TOP_N))

COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "10") or "10")

# совместимость: можно ставить ATR_THRESHOLD и MIN_PROBABILITY как раньше
BASE_ATR_THRESHOLD = float(os.getenv("BASE_ATR_THRESHOLD", os.getenv("ATR_THRESHOLD", "0.010")) or "0.010")  # в %
BASE_MIN_PROBABILITY = int(os.getenv("BASE_MIN_PROBABILITY", os.getenv("MIN_PROBABILITY", "60")) or "60")

ADAPTIVE_FILTERS = (os.getenv("ADAPTIVE_FILTERS", "1").strip() == "1")

NO_SIGNAL_NOTICE = (os.getenv("NO_SIGNAL_NOTICE", "1").strip() == "1")
NO_SIGNAL_COOLDOWN_MIN = int(os.getenv("NO_SIGNAL_COOLDOWN_MIN", "15") or "15")

# =========================
# ЛОГИ
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("impuls")

# =========================
# СОСТОЯНИЕ (в памяти)
# =========================
STATS = {
    "signals": 0,
    "win": 0,
    "loss": 0,
    "pulse_on": True,
    "last_signal": None,
    "last_no_signal_notice": None,
}

DAY = {
    "date": None,
    "signals": 0,
    "win": 0,
    "loss": 0,
}

LAST_SENT_BY_SYMBOL: Dict[str, datetime] = {}  # анти-спам по символу


# =========================
# УТИЛИТЫ
# =========================
def now_tz() -> datetime:
    return datetime.now(TZ)

def today_tz() -> date:
    return now_tz().date()

def ensure_day_rollover() -> None:
    d = today_tz()
    if DAY["date"] != d:
        DAY["date"] = d
        DAY["signals"] = 0
        DAY["win"] = 0
        DAY["loss"] = 0

def fmt_time(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%H:%M:%S")

def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

def direction_label(direction: str) -> str:
    return "⬆️ ВВЕРХ" if direction.upper() == "CALL" else "⬇️ ВНИЗ"

def require_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN пустой. Railway → Variables → BOT_TOKEN")
    if not TWELVE_API_KEY:
        raise RuntimeError("TWELVE_API_KEY пустой. Railway → Variables → TWELVE_API_KEY")
    if not CHANNEL_ID:
        raise RuntimeError("CHANNEL_ID пустой. Railway → Variables → CHANNEL_ID")
    if OWNER_ID == 0:
        log.warning("OWNER_ID = 0 (не задан). WIN/LOSS и owner-команды будут недоступны.")


# =========================
# TWELVE DATA
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

    if data.get("status") == "error":
        raise RuntimeError(f"TwelveData error for {symbol}: {data.get('message')}")

    values = data.get("values") or []
    if not values:
        raise RuntimeError(f"No candles returned for {symbol}")

    df = pd.DataFrame(values)
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
    return float((a / c) * 100.0)  # в %


# =========================
# АДАПТИВНЫЕ ФИЛЬТРЫ
# =========================
def compute_adaptive_thresholds(symbols: List[str]) -> Tuple[float, int, float]:
    """
    Возвращает (atr_threshold_pct, min_probability, median_atr_pct).
    Смысл: рынок стал тихий → пороги ослабляем, рынок стал активный → ужесточаем.
    """
    atrs: List[float] = []
    for s in symbols:
        try:
            df = td_time_series(s, TF, min(CANDLES, 180))
            atrs.append(atr_percent(df, 14))
        except Exception:
            continue

    if not atrs:
        return BASE_ATR_THRESHOLD, BASE_MIN_PROBABILITY, 0.0

    med = float(np.median(atrs))

    raw_thr = med * 0.80
    lo = BASE_ATR_THRESHOLD * 0.50
    hi = BASE_ATR_THRESHOLD * 2.00
    thr = max(lo, min(hi, raw_thr))

    ratio = med / max(BASE_ATR_THRESHOLD, 1e-6)
    adj = int(round((ratio - 1.0) * 6))
    adj = max(-8, min(8, adj))
    min_prob = max(55, min(85, BASE_MIN_PROBABILITY + adj))

    return thr, min_prob, med


# =========================
# СИГНАЛ
# =========================
@dataclass
class Signal:
    symbol: str
    direction: str           # CALL/PUT
    probability: int
    price: float
    rsi14: float
    ema50: float
    ema200: float
    atr14_pct: float
    entry_time: datetime
    exit_time: datetime
    reason: str

def compute_signal(symbol: str, atr_thr: float, min_prob_gate: int) -> Optional[Signal]:
    df = td_time_series(symbol, TF, CANDLES)

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
    reasons = []

    # Тренд (база)
    if trend_up:
        score += 35
        direction = "CALL"
        reasons.append("EMA50 > EMA200")
        # RSI для UP (ослаблено, чтобы бот не молчал)
        if 42 <= rsi_v <= 68:
            score += 30
            reasons.append("RSI ok (up)")
        else:
            score -= 10
            reasons.append("RSI weak (up)")
    elif trend_down:
        score += 35
        direction = "PUT"
        reasons.append("EMA50 < EMA200")
        # RSI для DOWN
        if 32 <= rsi_v <= 58:
            score += 30
            reasons.append("RSI ok (down)")
        else:
            score -= 10
            reasons.append("RSI weak (down)")
    else:
        return None

    # Сила тренда — бонус 0..15
    strength = abs(ema50_v - ema200_v) / max(close, 1e-9) * 100.0
    score += int(min(15, strength / 0.03))

    # Волатильность — бонус 0..15
    vol_ratio = atr_pct / max(atr_thr, 1e-6)
    score += int(min(15, max(0.0, (vol_ratio - 1.0) * 6)))

    probability = int(max(55, min(92, score)))
    if probability < min_prob_gate:
        return None

    entry = now_tz()
    exit_ = entry + timedelta(minutes=EXPIRY_MINUTES)

    reasons.append(f"ATR={atr_pct:.3f}% (thr={atr_thr:.3f}%)")

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

def pick_signals(symbols: List[str], atr_thr: float, min_prob_gate: int) -> List[Signal]:
    out: List[Signal] = []
    for s in symbols:
        last = LAST_SENT_BY_SYMBOL.get(s)
        if last and (now_tz() - last).total_seconds() < COOLDOWN_MINUTES * 60:
            continue

        try:
            sig = compute_signal(s, atr_thr=atr_thr, min_prob_gate=min_prob_gate)
        except Exception as e:
            log.warning("Signal error for %s: %s", s, e)
            continue

        if sig:
            out.append(sig)

    out.sort(key=lambda x: x.probability, reverse=True)
    return out


# =========================
# TELEGRAM: сообщения
# =========================
def signal_message(sig: Signal, atr_thr: float, min_prob_gate: int, med_atr: float) -> str:
    dir_text = direction_label(sig.direction)
    return (
        f"📊 *СИГНАЛ {sig.symbol}*\n"
        f"📈 Направление: *{dir_text}*\n"
        f"🔥 Вероятность: *{sig.probability}%*\n"
        f"⚡ ATR(14): `{sig.atr14_pct:.3f}%`\n\n"
        f"💰 Цена: `{sig.price:.5f}`\n"
        f"📉 RSI(14): `{sig.rsi14:.1f}`\n"
        f"📍 EMA50: `{sig.ema50:.5f}`\n"
        f"📍 EMA200: `{sig.ema200:.5f}`\n\n"
        f"⏱ Вход: *{fmt_time(sig.entry_time)}*\n"
        f"🏁 Выход: *{fmt_time(sig.exit_time)}*  (экспирация {EXPIRY_MINUTES} мин)\n"
        f"🌍 Таймзона: `{TIMEZONE_NAME}`\n"
        f"🧠 Фильтры: ATR_thr=`{atr_thr:.3f}%` | MIN_PROB=`{min_prob_gate}`"
        + (f" | median_ATR=`{med_atr:.3f}%`" if ADAPTIVE_FILTERS else "")
    )

def winloss_keyboard(signal_id: str, symbol: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ WIN", callback_data=f"wl|win|{signal_id}|{symbol}"),
        InlineKeyboardButton("❌ LOSS", callback_data=f"wl|loss|{signal_id}|{symbol}"),
    ]])

async def post_to_channel(context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None, reply_to: Optional[int] = None) -> None:
    await context.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
        disable_web_page_preview=True,
        reply_to_message_id=reply_to,
    )

def trade_report_text(symbol: str, result: str) -> str:
    ensure_day_rollover()
    w = DAY["win"]
    l = DAY["loss"]
    s = DAY["signals"]
    wr = (w / max(1, (w + l))) * 100.0
    return (
        f"📌 *ОТЧЁТ ПО СДЕЛКЕ*\n"
        f"Пара: *{symbol}*\n"
        f"Результат: *{result}*\n\n"
        f"📈 *Статистика за сегодня* ({DAY['date'].strftime('%d.%m.%Y')}, {TIMEZONE_NAME})\n"
        f"Сигналов: *{s}*\n"
        f"WIN: *{w}*\n"
        f"LOSS: *{l}*\n"
        f"WinRate: *{wr:.1f}%*"
    )


# =========================
# JOBS
# =========================
async def job_send_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_day_rollover()

    if ADAPTIVE_FILTERS:
        atr_thr, min_prob_gate, med_atr = compute_adaptive_thresholds(SYMBOLS)
    else:
        atr_thr, min_prob_gate, med_atr = BASE_ATR_THRESHOLD, BASE_MIN_PROBABILITY, 0.0

    candidates = pick_signals(SYMBOLS, atr_thr=atr_thr, min_prob_gate=min_prob_gate)

    if SEND_MODE == "ALL":
        to_send = candidates[:6]
    else:
        to_send = candidates[:TOP_N]

    if not to_send:
        if NO_SIGNAL_NOTICE:
            last = STATS.get("last_no_signal_notice")
            if not last or (now_tz() - last).total_seconds() >= NO_SIGNAL_COOLDOWN_MIN * 60:
                STATS["last_no_signal_notice"] = now_tz()
                await post_to_channel(
                    context,
                    f"🕵️ *{CHANNEL_NAME}*: сейчас нет сильных сигналов.\n"
                    f"Фильтры: ATR_thr=`{atr_thr:.3f}%` | MIN_PROB=`{min_prob_gate}`"
                    + (f" | median_ATR=`{med_atr:.3f}%`" if ADAPTIVE_FILTERS else "")
                )
        return

    for sig in to_send:
        signal_id = sig.entry_time.strftime("%Y%m%d%H%M%S")

        STATS["signals"] += 1
        DAY["signals"] += 1

        STATS["last_signal"] = {
            "id": signal_id,
            "symbol": sig.symbol,
            "time": sig.entry_time.isoformat(),
            "prob": sig.probability,
        }

        text = signal_message(sig, atr_thr=atr_thr, min_prob_gate=min_prob_gate, med_atr=med_atr)
        await post_to_channel(context, text, reply_markup=winloss_keyboard(signal_id, sig.symbol))

        LAST_SENT_BY_SYMBOL[sig.symbol] = now_tz()

async def job_pulse(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not STATS.get("pulse_on", True):
        return
    ensure_day_rollover()
    await post_to_channel(
        context,
        f"🕒 *{CHANNEL_NAME}*: бот жив, анализирую рынок…\n"
        f"Сегодня: signals={DAY['signals']} | win={DAY['win']} | loss={DAY['loss']}"
    )

async def job_daily_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_day_rollover()
    w = DAY["win"]
    l = DAY["loss"]
    s = DAY["signals"]
    wr = (w / max(1, w + l)) * 100.0

    txt = (
        f"📌 *{CHANNEL_NAME} — ЕЖЕДНЕВНЫЙ ОТЧЁТ*\n"
        f"🗓 Дата: *{DAY['date'].strftime('%d.%m.%Y')}*  ({TIMEZONE_NAME})\n\n"
        f"📨 Сигналов: *{s}*\n"
        f"✅ WIN: *{w}*\n"
        f"❌ LOSS: *{l}*\n"
        f"🎯 WinRate: *{wr:.1f}%*\n"
        f"⚙️ Режим: *{SEND_MODE}* (TOP_N={TOP_N})"
    )
    await post_to_channel(context, txt)


# =========================
# HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ IMPULS активен.\n\n"
        f"Канал: {CHANNEL_NAME}\n"
        f"Таймзона: {TIMEZONE_NAME}\n"
        f"Режим: {SEND_MODE} (TOP_N={TOP_N})\n"
        f"Adaptive: {'ON' if ADAPTIVE_FILTERS else 'OFF'}\n\n"
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
    ensure_day_rollover()
    w = DAY["win"]
    l = DAY["loss"]
    s = DAY["signals"]
    wr = (w / max(1, w + l)) * 100.0
    last = STATS.get("last_signal")

    await update.message.reply_text(
        "📊 Статистика\n"
        f"Сегодня ({DAY['date'].strftime('%d.%m.%Y')}): signals={s}, win={w}, loss={l}, WR={wr:.1f}%\n"
        f"Всего: signals={STATS['signals']}, win={STATS['win']}, loss={STATS['loss']}\n"
        f"Последний сигнал: {last}\n"
        f"Режим: {SEND_MODE} (TOP_N={TOP_N})\n"
        f"Adaptive: {'ON' if ADAPTIVE_FILTERS else 'OFF'}\n"
        f"TIMEZONE: {TIMEZONE_NAME}"
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

    parts = (q.data or "").split("|")
    if len(parts) != 4 or parts[0] != "wl":
        return

    action = parts[1]   # win / loss
    symbol = parts[3]

    ensure_day_rollover()

    if action == "win":
        STATS["win"] += 1
        DAY["win"] += 1
        result = "✅ WIN"
    elif action == "loss":
        STATS["loss"] += 1
        DAY["loss"] += 1
        result = "❌ LOSS"
    else:
        return

    # ✅ Авто-отчёт после каждой сделки: ответом к сообщению сигнала
    await post_to_channel(
        context,
        trade_report_text(symbol=symbol, result=result),
        reply_to=q.message.message_id
    )


# =========================
# MAIN
# =========================
def main() -> None:
    require_env()
    ensure_day_rollover()

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

    app.job_queue.run_repeating(job_send_signals, interval=SIGNAL_INTERVAL_SECONDS, first=10, name="signals")
    app.job_queue.run_repeating(job_pulse, interval=PULSE_INTERVAL_SECONDS, first=60, name="pulse")

    report_t = time(hour=REPORT_HOUR, minute=REPORT_MINUTE, tzinfo=TZ)
    app.job_queue.run_daily(job_daily_report, time=report_t, name="daily_report")

    log.info(
        "%s | started | TZ=%s | mode=%s TOP_N=%s | adaptive=%s | TF=%s",
        CHANNEL_NAME, TIMEZONE_NAME, SEND_MODE, TOP_N, ADAPTIVE_FILTERS, TF
    )

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
