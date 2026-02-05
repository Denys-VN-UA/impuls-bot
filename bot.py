# bot.py
# IMPULS ⚡ — TwelveData версия (FINAL)
# - TOP_N лучших сигналов
# - Не молчит (анти-спам)
# - Торговое время ПН–ПТ 10:00–20:00 (TZ), СБ/ВС выходной
# - ATR фильтр + (опционально) adaptive
# - Авто-отчёт после экспирации (без ID внизу)
# - ГИБРИД экспирации 3/5 минут по ATR (HYBRID_EXPIRY=1)

import os
import logging
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any

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
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()  # -100... или @channel
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "IMPULS ⚡")

TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Kyiv")
TZ = ZoneInfo(TIMEZONE_NAME)

# Сканер / интервалы
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "600"))  # 10 минут (экономит API)
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "600"))    # 10 минут
TF = os.getenv("TF", "1min")
CANDLES = int(os.getenv("CANDLES", "250"))

# Пары
SYMBOLS = os.getenv("SYMBOLS", "EUR/USD,USD/JPY,USD/CHF").split(",")

# Отправка сигналов
SEND_MODE = os.getenv("SEND_MODE", "TOP").strip().upper()   # TOP or BEST
TOP_N = int(os.getenv("TOP_N", "2"))
MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", "60"))

# ATR фильтр
ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.020"))  # в % (пример: 0.020 = 0.020%)
ADAPTIVE_FILTERS = os.getenv("ADAPTIVE_FILTERS", "0").strip() in ("1", "true", "True", "YES", "yes")
GLOBAL_ATR_MULT = float(os.getenv("GLOBAL_ATR_MULT", "1.00"))

# ГИБРИД экспирации
HYBRID_EXPIRY = os.getenv("HYBRID_EXPIRY", "1").strip() in ("1", "true", "True", "YES", "yes")
EXPIRY_FAST = int(os.getenv("EXPIRY_FAST", "3"))          # 3 минуты
EXPIRY_SLOW = int(os.getenv("EXPIRY_SLOW", "5"))          # 5 минут
HYBRID_ATR_BORDER = float(os.getenv("HYBRID_ATR_BORDER", "0.018"))  # если ATR% ниже — ставим 5 мин

# Анти-спам и паузы
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "6"))
WEAK_NOTICE_COOLDOWN_MINUTES = int(os.getenv("WEAK_NOTICE_COOLDOWN_MINUTES", "30"))
OFFHOURS_NOTICE_COOLDOWN_MINUTES = int(os.getenv("OFFHOURS_NOTICE_COOLDOWN_MINUTES", "60"))

# Старший тренд-фильтр (не обязателен)
TREND_FILTER = os.getenv("TREND_FILTER", "0").strip() in ("1", "true", "True", "YES", "yes")
TREND_TF = os.getenv("TREND_TF", "15min").strip()  # 15min (можно 5min/30min)

# Торговые часы (ПН–ПТ)
TRADE_START = os.getenv("TRADE_START", "10:00").strip()  # HH:MM
TRADE_END = os.getenv("TRADE_END", "20:00").strip()      # HH:MM

# Ежедневный отчёт (опционально)
REPORT_HOUR = int(os.getenv("REPORT_HOUR", "20"))
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", "0"))

# =========================
# ЛОГИ
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("impuls")
TD_BASE = "https://api.twelvedata.com"

# =========================
# ВНУТРЕННЯЯ СТАТИСТИКА
# =========================
STATS: Dict[str, Any] = {
    "day": None,
    "signals": 0,
    "win": 0,
    "loss": 0,
    "pulse_on": True,

    "last_signal": None,
    "last_weak_notice": None,
    "last_offhours_notice": None,

    "pair_cooldown_until": {},  # symbol -> datetime
}

def now_tz() -> datetime:
    return datetime.now(TZ)

def day_key(dt: Optional[datetime] = None) -> str:
    dt = dt or now_tz()
    return dt.strftime("%Y-%m-%d")

def ensure_day_reset() -> None:
    today = day_key()
    if STATS["day"] != today:
        STATS["day"] = today
        STATS["signals"] = 0
        STATS["win"] = 0
        STATS["loss"] = 0
        STATS["last_signal"] = None
        log.info("Day stats reset: %s", today)

def fmt_time(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%H:%M:%S")

def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

def require_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN пустой. Добавь BOT_TOKEN в Railway Variables.")
    if not TWELVE_API_KEY:
        raise RuntimeError("TWELVE_API_KEY пустой. Добавь TWELVE_API_KEY в Railway Variables.")
    if not CHANNEL_ID:
        raise RuntimeError("CHANNEL_ID пустой. Добавь CHANNEL_ID в Railway Variables.")
    if OWNER_ID == 0:
        log.warning("OWNER_ID не задан (0). Owner-only команды и WIN/LOSS будут недоступны.")

# =========================
# ТОРГОВОЕ ВРЕМЯ
# =========================
def parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm), tzinfo=TZ)

TRADE_START_T = parse_hhmm(TRADE_START)
TRADE_END_T = parse_hhmm(TRADE_END)

def is_trading_time(dt: Optional[datetime] = None) -> bool:
    dt = dt or now_tz()
    if dt.weekday() >= 5:  # СБ/ВС
        return False
    t = dt.timetz()
    return (t >= TRADE_START_T) and (t <= TRADE_END_T)

# =========================
# TWELVE DATA
# =========================
def td_time_series(symbol: str, interval: str, outputsize: int = 200) -> pd.DataFrame:
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
        raise RuntimeError(data.get("message", "TwelveData error"))

    values = data.get("values") or []
    if not values:
        raise RuntimeError("No candles returned")

    df = pd.DataFrame(values).iloc[::-1].reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df

def td_quote_price(symbol: str) -> float:
    url = f"{TD_BASE}/quote"
    params = {"symbol": symbol, "apikey": TWELVE_API_KEY, "format": "JSON"}
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    if data.get("status") == "error":
        raise RuntimeError(data.get("message", "TwelveData quote error"))
    price = data.get("close") or data.get("price")
    return float(price)

# =========================
# ИНДИКАТОРЫ
# =========================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
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
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    a = atr(df, period).iloc[-1]
    c = df["close"].iloc[-1]
    if c == 0 or pd.isna(a) or pd.isna(c):
        return 0.0
    return float((a / c) * 100.0)

# =========================
# СИГНАЛЫ
# =========================
@dataclass
class Signal:
    symbol: str
    direction: str           # CALL / PUT
    probability: int
    price: float
    rsi14: float
    ema50: float
    ema200: float
    atr14_pct: float
    expiry_minutes: int      # <-- гибрид тут
    entry_time: datetime
    exit_time: datetime
    signal_id: str

def direction_label(direction: str) -> str:
    return "⬆️ ВВЕРХ" if direction.upper() == "CALL" else "⬇️ ВНИЗ"

def per_symbol_atr_min(symbol: str) -> float:
    base = ATR_THRESHOLD * GLOBAL_ATR_MULT
    if "JPY" in symbol.upper():
        return max(base, 0.020)
    return base

def adaptive_atr_threshold(df: pd.DataFrame) -> float:
    a = atr(df, 14)
    c = df["close"].replace(0, np.nan)
    atrp = (a / c) * 100.0
    tail = atrp.dropna().tail(120)
    if len(tail) < 30:
        return 0.012
    med = float(tail.median())
    return max(0.012, med * 0.70)

def confirm_higher_trend(symbol: str) -> Optional[str]:
    df = td_time_series(symbol, TREND_TF, 250)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    e50 = float(df["ema50"].iloc[-1])
    e200 = float(df["ema200"].iloc[-1])
    if e50 > e200:
        return "CALL"
    if e50 < e200:
        return "PUT"
    return None

def choose_expiry(atr_pct: float) -> int:
    if not HYBRID_EXPIRY:
        return EXPIRY_FAST
    # если ATR низкий — рынок вялый → 5 минут чаще логичнее
    return EXPIRY_SLOW if atr_pct < HYBRID_ATR_BORDER else EXPIRY_FAST

def compute_signal(symbol: str) -> Optional[Signal]:
    df = td_time_series(symbol, TF, CANDLES)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)

    atr_pct = atr_percent(df, 14)

    # жёсткий запрет слишком низкого ATR
    if atr_pct < 0.012:
        return None

    thr = per_symbol_atr_min(symbol)
    if ADAPTIVE_FILTERS:
        thr = max(thr, adaptive_atr_threshold(df))

    if atr_pct < thr:
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
        ht = confirm_higher_trend(symbol)
        if ht is None or ht != direction:
            return None
        score += 10

    vol_bonus = min(20, int((atr_pct / max(thr, 0.0001)) * 6))
    score += vol_bonus

    probability = max(55, min(92, int(score)))
    if probability < MIN_PROBABILITY:
        return None

    expiry_minutes = choose_expiry(atr_pct)
    entry = now_tz()
    exit_ = entry + timedelta(minutes=expiry_minutes)
    signal_id = f"{entry.strftime('%Y%m%d%H%M%S')}_{symbol.replace('/', '')}"

    return Signal(
        symbol=symbol,
        direction=direction,
        probability=probability,
        price=close,
        rsi14=rsi_v,
        ema50=ema50_v,
        ema200=ema200_v,
        atr14_pct=atr_pct,
        expiry_minutes=expiry_minutes,
        entry_time=entry,
        exit_time=exit_,
        signal_id=signal_id,
    )

def pick_top_signals(symbols: List[str]) -> List[Signal]:
    out: List[Signal] = []
    for s in symbols:
        sym = s.strip()
        if not sym:
            continue

        cd_until = STATS["pair_cooldown_until"].get(sym)
        if cd_until and now_tz() < cd_until:
            continue

        try:
            sig = compute_signal(sym)
        except Exception as e:
            log.warning("Signal error for %s: %s", sym, e)
            continue

        if sig:
            out.append(sig)

    out.sort(key=lambda x: x.probability, reverse=True)
    if not out:
        return []

    if SEND_MODE == "BEST":
        return [out[0]]

    return out[:max(1, min(TOP_N, 5))]

# =========================
# TELEGRAM TEXTS
# =========================
def signal_text(sig: Signal) -> str:
    return (
        f"📊 *СИГНАЛ {sig.symbol}*\n"
        f"📈 Направление: *{direction_label(sig.direction)}*\n"
        f"🔥 Вероятность: *{sig.probability}%*\n\n"
        f"💰 Цена: `{sig.price:.5f}`\n"
        f"⚡ ATR(14): `{sig.atr14_pct:.3f}%`\n"
        f"📉 RSI(14): `{sig.rsi14:.1f}`\n"
        f"📍 EMA50: `{sig.ema50:.5f}`\n"
        f"📍 EMA200: `{sig.ema200:.5f}`\n\n"
        f"⏱ Вход: *{fmt_time(sig.entry_time)}*\n"
        f"🏁 Выход: *{fmt_time(sig.exit_time)}*  (эксп. {sig.expiry_minutes} мин)\n"
        f"🌍 Таймзона: `{TIMEZONE_NAME}`"
    )

def expiry_report_text(sig: Signal, exit_price: float) -> str:
    entry_price = sig.price
    diff = exit_price - entry_price
    pct = (diff / entry_price) * 100.0 if entry_price else 0.0

    went_up = exit_price > entry_price
    went_dir = "⬆️ ВВЕРХ" if went_up else "⬇️ ВНИЗ"

    predicted_up = sig.direction.upper() == "CALL"
    quote_win = (went_up and predicted_up) or ((not went_up) and (not predicted_up))
    quote_result = "✅ WIN" if quote_win else "❌ LOSS"

    # аккуратный отчёт БЕЗ ID внизу
    return (
        f"⏱ *Экспирация прошла по {sig.symbol}*\n"
        f"📈 График пошёл: *{went_dir}*\n"
        f"💰 Цена: `{entry_price:.5f}` → `{exit_price:.5f}`\n"
        f"{'✅' if quote_win else '❌'} По котировкам это *{quote_result.split()[-1]}*\n\n"
        f"👉 Если у Pocket Option итог отличается — отметь вручную *WIN/LOSS* под сигналом."
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
async def job_send_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_day_reset()

    if not is_trading_time():
        last = STATS.get("last_offhours_notice")
        now = now_tz()
        if (last is None) or ((now - last).total_seconds() >= OFFHOURS_NOTICE_COOLDOWN_MINUTES * 60):
            STATS["last_offhours_notice"] = now
            await post_to_channel(
                context,
                f"🌙 *Сейчас не торговое время.*\n"
                f"📅 Торгую ПН–ПТ\n"
                f"⏰ {TRADE_START}–{TRADE_END} (`{TIMEZONE_NAME}`)"
            )
        return

    sigs = pick_top_signals(SYMBOLS)

    if not sigs:
        last = STATS.get("last_weak_notice")
        now = now_tz()
        if (last is None) or ((now - last).total_seconds() >= WEAK_NOTICE_COOLDOWN_MINUTES * 60):
            STATS["last_weak_notice"] = now
            await post_to_channel(context, "📉 *Рынок слабый — сильных сигналов нет.*\nЯ продолжаю анализировать…")
        return

    for sig in sigs:
        STATS["signals"] += 1
        STATS["last_signal"] = {"symbol": sig.symbol, "time": fmt_time(sig.entry_time)}
        STATS["pair_cooldown_until"][sig.symbol] = now_tz() + timedelta(minutes=COOLDOWN_MINUTES)

        await post_to_channel(context, signal_text(sig), reply_markup=winloss_keyboard(sig.signal_id))

        async def _expiry_job(ctx: ContextTypes.DEFAULT_TYPE, s: Signal = sig) -> None:
            try:
                price_exit = td_quote_price(s.symbol)
            except Exception as e:
                log.warning("Expiry quote error for %s: %s", s.symbol, e)
                return
            await post_to_channel(ctx, expiry_report_text(s, price_exit))

        if context.job_queue:
            context.job_queue.run_once(_expiry_job, when=sig.exit_time)

async def job_pulse(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not STATS.get("pulse_on", True):
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
        f"📅 Дата: *{now_tz().strftime('%d.%m.%Y')}* (`{TIMEZONE_NAME}`)\n\n"
        f"✉️ Сигналов: *{s}*\n"
        f"✅ WIN: *{w}*\n"
        f"❌ LOSS: *{l}*\n"
        f"🎯 WinRate: *{wr:.1f}%*"
    )
    await post_to_channel(context, txt)

# =========================
# HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ Бот активен.\n"
        f"Канал: {CHANNEL_NAME}\n"
        f"Таймзона: {TIMEZONE_NAME}\n"
        f"Торговое время: ПН–ПТ {TRADE_START}–{TRADE_END}\n"
        f"Гибрид экспирации: {'ON' if HYBRID_EXPIRY else 'OFF'} (ATR<{HYBRID_ATR_BORDER:.3f}% → {EXPIRY_SLOW}м, иначе {EXPIRY_FAST}м)\n",
        disable_web_page_preview=True,
    )

async def test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    await post_to_channel(context, "✅ *ТЕСТ*: бот может писать в канал (OK)")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    ensure_day_reset()
    s = STATS["signals"]
    w = STATS["win"]
    l = STATS["loss"]
    wr = (w / max(1, w + l)) * 100.0
    last = STATS.get("last_signal") or "—"

    await update.message.reply_text(
        f"📊 Статистика (сегодня)\n"
        f"Сигналов: {s}\n"
        f"WIN: {w}\n"
        f"LOSS: {l}\n"
        f"WinRate: {wr:.1f}%\n"
        f"Последний: {last}\n"
        f"TZ: {TIMEZONE_NAME}",
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

    if not is_owner(q.from_user.id):
        await q.answer("⛔ Только владелец может отмечать WIN/LOSS.", show_alert=True)
        return

    ensure_day_reset()

    data = (q.data or "").split("|")
    if len(data) != 3 or data[0] != "wl":
        return

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

    report_t = dtime(hour=REPORT_HOUR, minute=REPORT_MINUTE, tzinfo=TZ)
    app.job_queue.run_daily(job_daily_report, time=report_t, name="daily_report")

    log.info("%s | started | TZ=%s | mode=%s top=%s | hybrid=%s border=%.3f",
             CHANNEL_NAME, TIMEZONE_NAME, SEND_MODE, TOP_N, HYBRID_EXPIRY, HYBRID_ATR_BORDER)

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
