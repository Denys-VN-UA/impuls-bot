# bot.py
# IMPULS ⚡ FINAL: never silent + TOP_N + adaptive ATR + cooldown
# python-telegram-bot[job-queue]==22.5

import os
import logging
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any, Tuple

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
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()  # обязателен
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Kyiv").strip()
TZ = ZoneInfo(TIMEZONE_NAME)

# Символы
SYMBOLS = os.getenv(
    "SYMBOLS",
    "USD/JPY,USD/CHF,EUR/USD,GBP/USD,EUR/JPY,GBP/JPY,AUD/USD,USD/CAD"
).split(",")

# Частоты
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "180"))  # 3 минуты
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "600"))    # 10 минут

# TF / свечи
TF = os.getenv("TF", "1min")
CANDLES = int(os.getenv("CANDLES", "250"))

# Экспирация (для текста)
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "3"))

# Фильтры
ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.020"))  # порог в % (например 0.020 = 0.020%)
ADAPTIVE_FILTERS = os.getenv("ADAPTIVE_FILTERS", "0").strip() == "1"

MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", os.getenv("MIN_PROB_TO_SEND", "75")))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))

# Режим отправки
# BEST = только лучшие TOP_N
# ALL  = отправлять все найденные (не рекомендую, но оставил)
SEND_MODE = os.getenv("SEND_MODE", "BEST").strip().upper()

# Сколько лучших отправлять
TOP_N = int(os.getenv("TOP_N", "1"))

# Чтобы “не молчал” — как часто можно писать “рынок слабый”
# (защита от спама)
NO_SIGNAL_COOLDOWN_MINUTES = int(os.getenv("NO_SIGNAL_COOLDOWN_MINUTES", "5"))

# TwelveData ограничение: если поймаем rate-limit — не спамим сообщениями
API_ERROR_COOLDOWN_MINUTES = int(os.getenv("API_ERROR_COOLDOWN_MINUTES", "10"))

TD_BASE = "https://api.twelvedata.com"

# =========================
# ЛОГИ
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("impuls")

# =========================
# СТАТИСТИКА / ПАМЯТЬ
# =========================
STATS = {
    "signals": 0,
    "win": 0,
    "loss": 0,
    "pulse_on": True,
    "last_signal": None,
}

LAST_SENT_BY_SYMBOL: Dict[str, datetime] = {}
LAST_NO_SIGNAL_MSG_AT: Optional[datetime] = None
LAST_API_ERROR_MSG_AT: Optional[datetime] = None


# =========================
# УТИЛИТЫ
# =========================
def now_tz() -> datetime:
    return datetime.now(TZ)

def fmt_time(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%H:%M:%S")

def direction_label(direction: str) -> str:
    return "⬆️ ВВЕРХ" if direction.upper() == "CALL" else "⬇️ ВНИЗ"

def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

def require_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN пустой. Добавь BOT_TOKEN в Railway → Variables.")
    if not TWELVE_API_KEY:
        raise RuntimeError("TWELVE_API_KEY пустой. Добавь TWELVE_API_KEY в Railway → Variables.")
    if not CHANNEL_ID:
        raise RuntimeError("CHANNEL_ID пустой. Добавь CHANNEL_ID в Railway → Variables.")

def cooldown_ok(symbol: str, minutes: int) -> bool:
    last = LAST_SENT_BY_SYMBOL.get(symbol)
    if not last:
        return True
    return (now_tz() - last).total_seconds() >= minutes * 60

def can_send_no_signal() -> bool:
    global LAST_NO_SIGNAL_MSG_AT
    if LAST_NO_SIGNAL_MSG_AT is None:
        return True
    return (now_tz() - LAST_NO_SIGNAL_MSG_AT).total_seconds() >= NO_SIGNAL_COOLDOWN_MINUTES * 60

def can_send_api_error() -> bool:
    global LAST_API_ERROR_MSG_AT
    if LAST_API_ERROR_MSG_AT is None:
        return True
    return (now_tz() - LAST_API_ERROR_MSG_AT).total_seconds() >= API_ERROR_COOLDOWN_MINUTES * 60


# =========================
# TWELVEDATA
# =========================
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
        msg = data.get("message", "Unknown TwelveData error")
        raise RuntimeError(msg)

    values = data.get("values") or []
    if not values:
        raise RuntimeError("No candles returned")

    df = pd.DataFrame(values)
    df = df.iloc[::-1].reset_index(drop=True)  # oldest -> newest

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


# =========================
# СИГНАЛ
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
    reason: str

def compute_signal(symbol: str, effective_atr_threshold: float) -> Optional[Signal]:
    df = td_time_series(symbol, TF, CANDLES)

    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)
    atr_pct = atr_percent(df, 14)

    # ATR фильтр
    if atr_pct < effective_atr_threshold:
        return None

    close = float(df["close"].iloc[-1])
    ema50_v = float(df["ema50"].iloc[-1])
    ema200_v = float(df["ema200"].iloc[-1])
    rsi_v = float(df["rsi14"].iloc[-1])

    trend_up = ema50_v > ema200_v
    trend_down = ema50_v < ema200_v

    score = 0
    reasons = []

    direction = None

    if trend_up:
        score += 35
        reasons.append("EMA50 > EMA200 (тренд вверх)")
        # мягче (чтобы не молчал)
        if 42 <= rsi_v <= 68:
            score += 35
            reasons.append("RSI подтверждает вверх")
            direction = "CALL"
        else:
            reasons.append("RSI слабый для вверх")

    elif trend_down:
        score += 35
        reasons.append("EMA50 < EMA200 (тренд вниз)")
        if 32 <= rsi_v <= 58:
            score += 35
            reasons.append("RSI подтверждает вниз")
            direction = "PUT"
        else:
            reasons.append("RSI слабый для вниз")
    else:
        return None

    if direction is None:
        return None

    # бонус за волатильность (чтобы 3 минуты “доходило”)
    vol_bonus = min(22, int((atr_pct / max(effective_atr_threshold, 0.0001)) * 6))
    score += vol_bonus
    reasons.append(f"ATR(14) {atr_pct:.3f}% (порог {effective_atr_threshold:.3f}%)")

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


# =========================
# ADAPTIVE ATR
# =========================
def calc_effective_atr_threshold(context: ContextTypes.DEFAULT_TYPE) -> Tuple[float, str]:
    """
    Если ADAPTIVE_FILTERS=1:
      - берём ATR% по всем символам (последние свечи)
      - ставим порог как max(ATR_THRESHOLD, percentile(ATR%, 40))
        => в тихом рынке порог не завышается, в бодром — не слишком низкий.
    Если выключено: возвращаем ATR_THRESHOLD.
    """
    if not ADAPTIVE_FILTERS:
        return ATR_THRESHOLD, f"fixed ATR={ATR_THRESHOLD:.3f}%"

    atrs: List[float] = []
    errors = 0

    for s in SYMBOLS:
        s = s.strip()
        if not s:
            continue
        try:
            df = td_time_series(s, TF, min(CANDLES, 160))
            atrs.append(atr_percent(df, 14))
        except Exception:
            errors += 1

    if not atrs:
        # если совсем нет данных — fallback на фиксированный
        return ATR_THRESHOLD, f"adaptive fallback (no data), ATR={ATR_THRESHOLD:.3f}%"

    p40 = float(np.percentile(atrs, 40))
    # не опускаем ниже 60% базового, чтобы не ловить мусор,
    # но и не делаем порог слишком жёстким
    floor = ATR_THRESHOLD * 0.60
    effective = max(floor, min(ATR_THRESHOLD * 1.50, p40))

    info = f"adaptive ATR={effective:.3f}% (p40={p40:.3f}%, base={ATR_THRESHOLD:.3f}%, errors={errors})"
    return effective, info


# =========================
# TELEGRAM: текст/кнопки
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
    )

def winloss_keyboard(signal_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ WIN", callback_data=f"wl|win|{signal_id}"),
        InlineKeyboardButton("❌ LOSS", callback_data=f"wl|loss|{signal_id}"),
    ]])

async def post_to_channel(context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None) -> None:
    await context.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )


# =========================
# JOB: СИГНАЛЫ (TOP_N + never silent)
# =========================
async def job_send_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    global LAST_NO_SIGNAL_MSG_AT, LAST_API_ERROR_MSG_AT

    effective_atr, atr_info = calc_effective_atr_threshold(context)

    signals: List[Signal] = []
    rejected_low_prob = 0
    rejected_cooldown = 0
    scanned = 0

    for s in SYMBOLS:
        symbol = s.strip()
        if not symbol:
            continue
        scanned += 1

        # cooldown на символ
        if not cooldown_ok(symbol, COOLDOWN_MINUTES):
            rejected_cooldown += 1
            continue

        try:
            sig = compute_signal(symbol, effective_atr)
        except Exception as e:
            msg = str(e).lower()

            # TwelveData rate limit/credits
            if ("run out of api credits" in msg) or ("rate limit" in msg) or ("limit" in msg and "credits" in msg):
                if can_send_api_error():
                    LAST_API_ERROR_MSG_AT = now_tz()
                    await post_to_channel(
                        context,
                        "⚠️ *Данные временно недоступны (лимит API).* \n"
                        "Я не молчу — просто провайдер ограничил запросы. Попробуй позже или уменьши частоту/список пар."
                    )
                return

            # прочие ошибки — просто лог
            log.warning("Signal error for %s: %s", symbol, e)
            continue

        if not sig:
            continue

        if sig.probability < MIN_PROBABILITY:
            rejected_low_prob += 1
            continue

        signals.append(sig)

    # сортируем по вероятности (лучшие вверху)
    signals.sort(key=lambda x: x.probability, reverse=True)

    if SEND_MODE == "ALL":
        to_send = signals
    else:
        # BEST mode
        to_send = signals[:max(1, TOP_N)]

    # Если нечего отправлять — НЕ МОЛЧИМ
    if not to_send:
        if can_send_no_signal():
            LAST_NO_SIGNAL_MSG_AT = now_tz()
            await post_to_channel(
                context,
                "😴 *Рынок слабый — сильных сетапов нет.*\n"
                f"• Сканировал: `{scanned}` пар\n"
                f"• Cooldown пропущено: `{rejected_cooldown}`\n"
                f"• Слабая вероятность (<{MIN_PROBABILITY}%): `{rejected_low_prob}`\n"
                f"• Фильтр: `{atr_info}`\n"
                "Я продолжаю мониторинг…"
            )
        return

    # Отправляем TOP_N (или ALL)
    for sig in to_send:
        STATS["signals"] += 1
        signal_id = sig.entry_time.strftime("%Y%m%d%H%M%S") + "_" + sig.symbol.replace("/", "")
        STATS["last_signal"] = {"id": signal_id, "symbol": sig.symbol, "at": sig.entry_time.isoformat()}

        await post_to_channel(context, signal_message(sig), reply_markup=winloss_keyboard(signal_id))
        LAST_SENT_BY_SYMBOL[sig.symbol] = now_tz()


# =========================
# JOB: ПУЛЬС
# =========================
async def job_pulse(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not STATS.get("pulse_on", True):
        return
    await post_to_channel(context, "🕒 *IMPULS*: бот жив, анализирую рынок…")


# =========================
# HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ Бот активен.\n\n"
        f"🌍 Таймзона: {TIMEZONE_NAME}\n"
        f"🔧 SEND_MODE: {SEND_MODE}\n"
        f"🔝 TOP_N: {TOP_N}\n"
        f"⚡ ATR_THRESHOLD: {ATR_THRESHOLD}\n"
        f"🧠 ADAPTIVE_FILTERS: {'ON' if ADAPTIVE_FILTERS else 'OFF'}\n\n"
        "Команды (только владелец):\n"
        "/test /stats /pulse_on /pulse_off\n",
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

    await update.message.reply_text(
        f"📊 Статистика\n"
        f"Сигналов: {s}\n"
        f"WIN: {w}\n"
        f"LOSS: {l}\n"
        f"WinRate: {wr:.1f}%\n"
        f"Последний: {last}\n"
        f"SEND_MODE={SEND_MODE}, TOP_N={TOP_N}\n"
        f"ADAPTIVE={'ON' if ADAPTIVE_FILTERS else 'OFF'}",
    )

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

    if not is_owner(q.from_user.id):
        await q.answer("⛔ Только владелец может отмечать WIN/LOSS.", show_alert=True)
        return

    parts = (q.data or "").split("|")
    if len(parts) != 3 or parts[0] != "wl":
        return

    action = parts[1]
    signal_id = parts[2]

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

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("test", test_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("pulse_on", pulse_on))
    app.add_handler(CommandHandler("pulse_off", pulse_off))
    app.add_handler(CallbackQueryHandler(on_button))

    if app.job_queue is None:
        raise RuntimeError("JobQueue не активен. Установи python-telegram-bot[job-queue]==22.5")

    # Сканер (никогда не молчит)
    app.job_queue.run_repeating(job_send_signals, interval=SIGNAL_INTERVAL_SECONDS, first=10)

    # Пульс
    app.job_queue.run_repeating(job_pulse, interval=PULSE_INTERVAL_SECONDS, first=60)

    log.info("IMPULS started | TF=%s | TOP_N=%s | SEND_MODE=%s | ADAPTIVE=%s",
             TF, TOP_N, SEND_MODE, ADAPTIVE_FILTERS)

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
