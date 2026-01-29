# bot.py
# IMPULS ⚡ — TwelveData версия
# ✅ TOP_N (1/2/3) лучших сигналов
# ✅ Никогда не молчит (пишет "рынок слабый" / "лимит API")
# ✅ ADAPTIVE_FILTERS=1 (адаптивный порог ATR)
# ✅ Авто-проверка после экспирации: бот сам пишет, куда пошёл график (по котировкам)
# python-telegram-bot[job-queue]==22.5

import os
import logging
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time
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

# Канал: "-100xxxxxxxxxx" или "@channel_username"
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()

OWNER_ID = int(os.getenv("OWNER_ID", "0"))
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "IMPULS ⚡")

TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Kyiv")
TZ = ZoneInfo(TIMEZONE_NAME)

# Частота сканера (совет для free TwelveData: 600 сек и 1–2 пары)
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "600"))

# Пульс (чтобы видеть что жив)
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "600"))
PULSE_ENABLED_DEFAULT = os.getenv("PULSE_ENABLED", "1").strip() == "1"

# Таймфрейм и свечи
TF = os.getenv("TF", "1min").strip()
CANDLES = int(os.getenv("CANDLES", "250"))

# Экспирация
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "3"))
EVAL_EXTRA_SECONDS = int(os.getenv("EVAL_EXTRA_SECONDS", "10"))

# Список пар
DEFAULT_SYMBOLS = [
    s.strip() for s in os.getenv(
        "SYMBOLS",
        "EUR/USD,USD/JPY"
    ).split(",")
    if s.strip()
]

# Сколько сигналов слать за цикл
TOP_N = int(os.getenv("TOP_N", "1"))  # 1/2/3

# Режим отправки:
# BEST = отправить только лучшие TOP_N
# ALL  = отправить ТОП-результаты, но может быть шумнее (всё равно ограничено TOP_N)
SEND_MODE = os.getenv("SEND_MODE", "BEST").strip().upper()

# Фильтры
MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", "70"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))

# ATR фильтр (в процентах)
ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.020"))  # 0.020%

# Адаптивный фильтр
ADAPTIVE_FILTERS = os.getenv("ADAPTIVE_FILTERS", "0").strip() == "1"
ADAPTIVE_LOOKBACK = int(os.getenv("ADAPTIVE_LOOKBACK", "60"))  # сколько последних баров смотреть
ADAPTIVE_ATR_MULT = float(os.getenv("ADAPTIVE_ATR_MULT", "1.0"))  # множитель к медиане ATR%

# Ежедневный отчёт (по желанию)
REPORT_HOUR = int(os.getenv("REPORT_HOUR", "22"))
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", "0"))

# =========================
# ЛОГИ
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("impuls")

# =========================
# СТАТЫ / ПАМЯТЬ
# =========================
STATS = {
    "signals": 0,
    "win": 0,
    "loss": 0,
    "pulse_on": PULSE_ENABLED_DEFAULT,
    "last_signal_id": None,
}

# анти-спам
LAST_SENT: Dict[str, datetime] = {}        # pair -> dt
LAST_NO_SIGNAL: Optional[datetime] = None  # чтобы не спамить "рынок слабый"
LAST_API_LIMIT: Optional[datetime] = None  # чтобы не спамить "лимит API"

# хранение сигналов для пост-оценки
SIGNALS: Dict[str, Dict[str, Any]] = {}  # signal_id -> data

# =========================
# ВСПОМОГАТЕЛЬНОЕ
# =========================
def now_tz() -> datetime:
    return datetime.now(TZ)

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

def direction_label(direction: str) -> str:
    return "⬆️ ВВЕРХ" if direction.upper() == "CALL" else "⬇️ ВНИЗ"

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# =========================
# TWELVEDATA
# =========================
TD_BASE = "https://api.twelvedata.com"

class RateLimitError(RuntimeError):
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
        msg = (data.get("message") or "TwelveData error").strip()
        low = msg.lower()
        if "api credits" in low or "run out" in low or "limit" in low or "rate" in low:
            raise RateLimitError(msg)
        raise RuntimeError(msg)

    values = data.get("values") or []
    if not values:
        raise RuntimeError(f"No candles returned for {symbol}")

    df = pd.DataFrame(values)
    # приходит newest->oldest, разворачиваем
    df = df.iloc[::-1].reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df

def td_last_price(symbol: str) -> float:
    # берём 2 свечи, чтобы точно была последняя
    df = td_time_series(symbol, TF, 2)
    return float(df["close"].iloc[-1])

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

def atr_percent_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    a = atr(df, period)
    c = df["close"].replace(0, np.nan)
    return (a / c) * 100.0

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

def compute_signal(symbol: str) -> Optional[Signal]:
    df = td_time_series(symbol, TF, CANDLES)

    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)

    atr_pct_s = atr_percent_series(df, 14)
    atr_pct = float(atr_pct_s.iloc[-1]) if pd.notna(atr_pct_s.iloc[-1]) else 0.0

    # ----- порог ATR (обычный или адаптивный) -----
    threshold = ATR_THRESHOLD
    if ADAPTIVE_FILTERS:
        tail = atr_pct_s.dropna().tail(max(20, ADAPTIVE_LOOKBACK))
        if len(tail) >= 10:
            med = float(tail.median())
            threshold = max(ATR_THRESHOLD, med * ADAPTIVE_ATR_MULT)

    if atr_pct < threshold:
        return None

    close = float(df["close"].iloc[-1])
    ema50_v = float(df["ema50"].iloc[-1])
    ema200_v = float(df["ema200"].iloc[-1])
    rsi_v = float(df["rsi14"].iloc[-1])

    trend_up = ema50_v > ema200_v
    trend_down = ema50_v < ema200_v

    direction = None
    score = 0.0
    reasons = []

    # тренд
    if trend_up:
        score += 35
        reasons.append("EMA50>EMA200 (вверх)")
        # RSI для импульса вверх
        if 45 <= rsi_v <= 65:
            score += 35
            direction = "CALL"
            reasons.append("RSI подтверждает вверх")
        else:
            reasons.append("RSI не подтверждает вверх")
    elif trend_down:
        score += 35
        reasons.append("EMA50<EMA200 (вниз)")
        # RSI для импульса вниз
        if 35 <= rsi_v <= 55:
            score += 35
            direction = "PUT"
            reasons.append("RSI подтверждает вниз")
        else:
            reasons.append("RSI не подтверждает вниз")
    else:
        return None

    if direction is None:
        return None

    # бонус волатильности
    # чем больше ATR% относительно threshold, тем больше бонус
    rel = atr_pct / max(threshold, 1e-6)
    vol_bonus = clamp((rel - 1.0) * 20.0, 0.0, 20.0)
    score += vol_bonus
    reasons.append(f"ATR(14)={atr_pct:.3f}% (thr={threshold:.3f}%)")

    probability = int(clamp(score + 20, 55, 92))

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

def pick_top_signals(symbols: List[str], top_n: int) -> List[Signal]:
    out: List[Signal] = []
    for s in symbols:
        s = s.strip()
        if not s:
            continue
        try:
            sig = compute_signal(s)
        except RateLimitError:
            # пробросим, чтобы обработать в job единым сообщением
            raise
        except Exception as e:
            log.warning("Signal error for %s: %s", s, e)
            continue

        if sig:
            out.append(sig)

    out.sort(key=lambda x: x.probability, reverse=True)
    return out[:max(1, min(10, top_n))]

# =========================
# TELEGRAM
# =========================
def winloss_keyboard(signal_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ WIN", callback_data=f"wl|win|{signal_id}"),
            InlineKeyboardButton("❌ LOSS", callback_data=f"wl|loss|{signal_id}"),
        ]
    ])

def signal_message(sig: Signal, signal_id: str) -> str:
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
        f"🆔 id: `{signal_id}`\n"
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
# POST-ОЦЕНКА ПОСЛЕ ЭКСПИРАЦИИ
# =========================
async def job_after_expiry(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data or {}
    signal_id = data.get("signal_id")
    if not signal_id or signal_id not in SIGNALS:
        return

    sig = SIGNALS[signal_id]
    symbol = sig["symbol"]
    direction = sig["direction"]
    entry_price = sig["entry_price"]

    try:
        exit_price = td_last_price(symbol)
    except RateLimitError:
        # если снова лимит — не спамим каждую минуту
        global LAST_API_LIMIT
        now = now_tz()
        if LAST_API_LIMIT is None or (now - LAST_API_LIMIT).total_seconds() > 1800:
            LAST_API_LIMIT = now
            await post_to_channel(
                context,
                "⚠️ *Данные временно недоступны (лимит API).* \n"
                "Я не молчу — просто провайдер ограничил запросы.\n"
                "Попробуй позже или уменьши частоту/список пар.",
            )
        return
    except Exception as e:
        log.warning("after_expiry error %s: %s", symbol, e)
        return

    move_up = exit_price > entry_price
    move = "⬆️ ВВЕРХ" if move_up else "⬇️ ВНИЗ"

    # "если бы ставили по сигналу" (аккуратно формулируем)
    would_be_win = (direction == "CALL" and move_up) or (direction == "PUT" and not move_up)
    verdict = "✅ *По котировкам это WIN* (движение в сторону сигнала)" if would_be_win else "❌ *По котировкам это LOSS* (движение против сигнала)"

    txt = (
        f"⏱ *Экспирация прошла по {symbol}*\n"
        f"📈 Движение графика: *{move}*\n"
        f"💰 Цена: `{entry_price:.5f}` → `{exit_price:.5f}`\n"
        f"{verdict}\n\n"
        f"👉 Отметь фактический результат кнопкой WIN/LOSS под сигналом.\n"
        f"🆔 id: `{signal_id}`"
    )
    await post_to_channel(context, txt)

# =========================
# JOB: СИГНАЛЫ
# =========================
async def job_send_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    global LAST_NO_SIGNAL, LAST_API_LIMIT

    now = now_tz()

    # собираем пары, пропуская те, что на cooldown
    symbols = []
    for s in DEFAULT_SYMBOLS:
        last = LAST_SENT.get(s)
        if last and (now - last).total_seconds() < COOLDOWN_MINUTES * 60:
            continue
        symbols.append(s)

    if not symbols:
        return

    try:
        top = pick_top_signals(symbols, TOP_N)
    except RateLimitError:
        # не спамим каждую минуту
        if LAST_API_LIMIT is None or (now - LAST_API_LIMIT).total_seconds() > 1800:
            LAST_API_LIMIT = now
            await post_to_channel(
                context,
                "⚠️ *Данные временно недоступны (лимит API).* \n"
                "Я не молчу — просто провайдер ограничил запросы.\n"
                "✅ Решение для TwelveData Free:\n"
                "• поставь `SIGNAL_INTERVAL_SECONDS=600`\n"
                "• оставь `SYMBOLS=EUR/USD,USD/JPY` (1–2 пары)\n",
            )
        return

    # если нет сигналов — тоже пишем, но редко
    if not top:
        if LAST_NO_SIGNAL is None or (now - LAST_NO_SIGNAL).total_seconds() > 900:
            LAST_NO_SIGNAL = now
            await post_to_channel(
                context,
                "📉 *Рынок слабый — сильных сигналов нет.*\n"
                "Я продолжаю анализировать…",
            )
        return

    # фильтр по MIN_PROBABILITY
    top = [s for s in top if s.probability >= MIN_PROBABILITY]
    if not top:
        if LAST_NO_SIGNAL is None or (now - LAST_NO_SIGNAL).total_seconds() > 900:
            LAST_NO_SIGNAL = now
            await post_to_channel(
                context,
                f"📉 *Сигналы есть, но ниже порога {MIN_PROBABILITY}%.*\n"
                "Я жду более сильные…",
            )
        return

    # SEND_MODE=BEST/ALL — по факту оба отправляют TOP_N, но BEST можно сделать 1 шт.
    to_send = top[:TOP_N] if SEND_MODE in ("ALL", "BEST") else top[:TOP_N]

    for sig in to_send:
        STATS["signals"] += 1

        # уникальный id
        sid = f"{sig.entry_time.strftime('%Y%m%d%H%M%S')}_{sig.symbol.replace('/', '')}"
        STATS["last_signal_id"] = sid

        # сохраняем для пост-оценки
        SIGNALS[sid] = {
            "symbol": sig.symbol,
            "direction": sig.direction,
            "entry_price": sig.price,
            "entry_time": sig.entry_time,
            "exit_time": sig.exit_time,
        }

        # отправка
        await post_to_channel(context, signal_message(sig, sid), reply_markup=winloss_keyboard(sid))

        # ставим cooldown
        LAST_SENT[sig.symbol] = now

        # планируем пост после экспирации
        delay = max(5, int((sig.exit_time - now_tz()).total_seconds()) + EVAL_EXTRA_SECONDS)
        context.job_queue.run_once(job_after_expiry, when=delay, data={"signal_id": sid}, name=f"expiry_{sid}")

# =========================
# JOB: ПУЛЬС
# =========================
async def job_pulse(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not STATS.get("pulse_on", True):
        return
    await post_to_channel(context, f"🕒 *{CHANNEL_NAME}*: бот жив, анализирую рынок…")

# =========================
# ЕЖЕДНЕВНЫЙ ОТЧЁТ (опционально)
# =========================
async def job_daily_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    w = STATS["win"]
    l = STATS["loss"]
    s = STATS["signals"]
    wr = (w / max(1, w + l)) * 100.0
    txt = (
        f"📌 *{CHANNEL_NAME} — ЕЖЕДНЕВНЫЙ ОТЧЁТ*\n"
        f"🗓 Дата: *{now_tz().strftime('%d.%m.%Y')}*  ({TIMEZONE_NAME})\n\n"
        f"📨 Сигналов: *{s}*\n"
        f"✅ WIN: *{w}*\n"
        f"❌ LOSS: *{l}*\n"
        f"🎯 WinRate: *{wr:.1f}%*\n"
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
        f"Пары: {', '.join(DEFAULT_SYMBOLS)}\n"
        f"TOP_N: {TOP_N} | SEND_MODE: {SEND_MODE}\n"
        f"ADAPTIVE_FILTERS: {'ON' if ADAPTIVE_FILTERS else 'OFF'}\n\n"
        "Команды (только владелец):\n"
        "/test — тест в канал\n"
        "/stats — статистика\n"
        "/pulse_on — включить пульс\n"
        "/pulse_off — выключить пульс\n"
        "/report_now — отчёт сейчас\n",
        disable_web_page_preview=True,
    )

async def test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    await post_to_channel(context, "✅ *ТЕСТ*: бот может писать в канал (OK)")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    w = STATS["win"]
    l = STATS["loss"]
    s = STATS["signals"]
    wr = (w / max(1, w + l)) * 100.0
    await update.message.reply_text(
        f"📊 Статистика\n"
        f"Сигналов: {s}\n"
        f"WIN: {w}\n"
        f"LOSS: {l}\n"
        f"WinRate: {wr:.1f}%\n"
        f"last_id: {STATS.get('last_signal_id')}\n"
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

    if not q or not q.data:
        return

    user_id = q.from_user.id
    if not is_owner(user_id):
        await q.answer("⛔ Только владелец может отмечать WIN/LOSS.", show_alert=True)
        return

    parts = q.data.split("|")
    if len(parts) != 3 or parts[0] != "wl":
        return

    action = parts[1]
    signal_id = parts[2]

    if action == "win":
        STATS["win"] += 1
        await q.message.reply_text(f"✅ WIN отмечен\n🆔 id: `{signal_id}`", parse_mode=ParseMode.MARKDOWN)
    elif action == "loss":
        STATS["loss"] += 1
        await q.message.reply_text(f"❌ LOSS отмечен\n🆔 id: `{signal_id}`", parse_mode=ParseMode.MARKDOWN)

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

    # Сканер
    app.job_queue.run_repeating(job_send_signals, interval=SIGNAL_INTERVAL_SECONDS, first=10, name="signals")

    # Пульс
    app.job_queue.run_repeating(job_pulse, interval=PULSE_INTERVAL_SECONDS, first=60, name="pulse")

    # Ежедневный отчёт (по желанию)
    report_t = time(hour=REPORT_HOUR, minute=REPORT_MINUTE, tzinfo=TZ)
    app.job_queue.run_daily(job_daily_report, time=report_t, name="daily_report")

    log.info(
        "%s | started | TZ=%s | symbols=%s | interval=%ss | TOP_N=%s | SEND_MODE=%s | adaptive=%s",
        CHANNEL_NAME, TIMEZONE_NAME, DEFAULT_SYMBOLS, SIGNAL_INTERVAL_SECONDS, TOP_N, SEND_MODE, ADAPTIVE_FILTERS
    )

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
