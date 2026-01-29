# bot.py
# IMPULS ⚡ — TwelveData версия (TOP_N + не молчит + лимит-защита + adaptive ATR)
# python-telegram-bot[job-queue]==22.5

import os
import time as pytime
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
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# =========================
# ENV
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "").strip()
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()  # -100xxxx or @channel
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "IMPULS ⚡")

TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Kyiv")
TZ = ZoneInfo(TIMEZONE_NAME)

# Частота сканирования
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "600"))  # 10 минут по умолчанию

# Сколько лучших пар отправлять
TOP_N = int(os.getenv("TOP_N", "1"))  # 1/2/3
TOP_N = max(1, min(3, TOP_N))

# Режим отправки:
# BEST = отправляет только 1 самую лучшую (игнорирует TOP_N)
# TOP  = отправляет TOP_N лучших (если прошли порог)
SEND_MODE = os.getenv("SEND_MODE", "TOP").strip().upper()
if SEND_MODE not in ("BEST", "TOP"):
    SEND_MODE = "TOP"

# Пары (ВАЖНО: на бесплатном TwelveData ставь 2 пары!)
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "EUR/USD,USD/JPY").split(",") if s.strip()]

# Таймфрейм и свечи
TF = os.getenv("TF", "1min")
CANDLES = int(os.getenv("CANDLES", "250"))

# Экспирация (мин)
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "3"))

# Порог вероятности, чтобы отправлять
MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", "70"))

# Кулдаун по паре (мин) — чтобы не спамить одной и той же
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "10"))

# ATR порог (%)
ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.020"))  # в % (0.020% это норм для форекса)

# Адаптивный режим
ADAPTIVE_FILTERS = os.getenv("ADAPTIVE_FILTERS", "0").strip() in ("1", "true", "True", "YES", "yes")

# Пульс
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "900"))  # 15 минут
PULSE_ON_DEFAULT = os.getenv("PULSE_ON", "1").strip() in ("1", "true", "True", "YES", "yes")

# Ограничение по API (на бесплатном часто 8 credits/min)
TD_MAX_CALLS_PER_MIN = int(os.getenv("TD_MAX_CALLS_PER_MIN", "8"))

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
# ПАМЯТЬ/СТАТЫ
# =========================
STATS = {
    "signals": 0,
    "win": 0,
    "loss": 0,
    "pulse_on": PULSE_ON_DEFAULT,
    "last_signal": None,
}

LAST_SENT_BY_SYMBOL: Dict[str, datetime] = {}
LAST_NO_SIGNAL_NOTICE: Optional[datetime] = None
LAST_API_LIMIT_NOTICE: Optional[datetime] = None

# Кеш свечей, чтобы не дергать API чаще чем нужно
CANDLE_CACHE: Dict[Tuple[str, str], Tuple[datetime, pd.DataFrame]] = {}  # (symbol, tf) -> (ts, df)
CACHE_TTL_SECONDS = 55  # если вызвали повторно в течение минуты — отдаем кеш

# Простой лимитер вызовов (окно 60 сек)
TD_CALL_TIMES: List[float] = []  # timestamps (epoch seconds)

# =========================
# УТИЛИТЫ
# =========================
def now_tz() -> datetime:
    return datetime.now(TZ)

def fmt_time(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%H:%M:%S")

def fmt_dt(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%d.%m.%Y %H:%M:%S")

def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

def direction_label(direction: str) -> str:
    # без CALL/PUT, только стрелки как ты просил
    if direction.upper() == "CALL":
        return "⬆️ ВВЕРХ"
    return "⬇️ ВНИЗ"

def require_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN пустой. Добавь BOT_TOKEN в Railway Variables.")
    if not TWELVE_API_KEY:
        raise RuntimeError("TWELVE_API_KEY пустой. Добавь TWELVE_API_KEY в Railway Variables.")
    if not CHANNEL_ID:
        raise RuntimeError("CHANNEL_ID пустой. Добавь CHANNEL_ID в Railway Variables.")
    if not SYMBOLS:
        raise RuntimeError("SYMBOLS пустой. Добавь SYMBOLS, например: EUR/USD,USD/JPY")

# =========================
# TwelveData
# =========================
TD_BASE = "https://api.twelvedata.com"

class TwelveLimitError(RuntimeError):
    pass

def _td_rate_ok() -> bool:
    """Не даём сделать больше TD_MAX_CALLS_PER_MIN запросов за 60 секунд."""
    global TD_CALL_TIMES
    now = pytime.time()
    TD_CALL_TIMES = [t for t in TD_CALL_TIMES if now - t < 60]
    return len(TD_CALL_TIMES) < TD_MAX_CALLS_PER_MIN

def _td_mark_call():
    TD_CALL_TIMES.append(pytime.time())

def td_time_series(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    # cache
    cache_key = (symbol, interval)
    cached = CANDLE_CACHE.get(cache_key)
    if cached:
        ts, df = cached
        if (now_tz() - ts).total_seconds() <= CACHE_TTL_SECONDS:
            return df.copy()

    if not _td_rate_ok():
        raise TwelveLimitError("Local limiter: too many requests per minute")

    url = f"{TD_BASE}/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_API_KEY,
        "format": "JSON",
        "timezone": "UTC",
    }

    _td_mark_call()
    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    if data.get("status") == "error":
        msg = (data.get("message") or "").lower()
        # TwelveData часто пишет про credits / limit
        if "credits" in msg or "limit" in msg or "too many" in msg:
            raise TwelveLimitError(data.get("message") or "TwelveData rate limit")
        raise RuntimeError(f"TwelveData error for {symbol}: {data.get('message')}")

    values = data.get("values") or []
    if not values:
        raise RuntimeError(f"No candles returned for {symbol}")

    df = pd.DataFrame(values)
    # values идут latest->oldest
    df = df.iloc[::-1].reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])

    CANDLE_CACHE[cache_key] = (now_tz(), df.copy())
    return df

# =========================
# Индикаторы
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

def atr_pct_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    a = atr(df, period)
    c = df["close"].replace(0, np.nan)
    return (a / c) * 100.0

def adaptive_atr_threshold(df: pd.DataFrame, base_thr: float) -> float:
    """
    Адаптивный порог:
    берём распределение ATR% за историю и ставим порог на нижний квантиль,
    чтобы в тихом рынке порог сам падал, а в шумном — рос.
    """
    s = atr_pct_series(df, 14).dropna()
    if len(s) < 50:
        return base_thr
    q = float(np.nanpercentile(s.values, 35))  # 35-й перцентиль
    # не даём уйти слишком низко/высоко
    low = base_thr * 0.6
    high = base_thr * 1.8
    return max(low, min(high, q))

# =========================
# Сигнал
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
    id: str

def compute_signal(symbol: str) -> Optional[Signal]:
    df = td_time_series(symbol, TF, CANDLES)

    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)
    atrp = atr_pct_series(df, 14).iloc[-1]
    atrp = float(atrp) if pd.notna(atrp) else 0.0

    thr = ATR_THRESHOLD
    if ADAPTIVE_FILTERS:
        thr = adaptive_atr_threshold(df, ATR_THRESHOLD)

    if atrp < thr:
        return None

    close = float(df["close"].iloc[-1])
    ema50_v = float(df["ema50"].iloc[-1])
    ema200_v = float(df["ema200"].iloc[-1])
    rsi_v = float(df["rsi14"].iloc[-1])

    trend_up = ema50_v > ema200_v
    trend_down = ema50_v < ema200_v
    if not (trend_up or trend_down):
        return None

    # Скоринг (простая, стабильная логика)
    score = 0
    reasons = []

    # тренд
    score += 35
    reasons.append("EMA50>EMA200" if trend_up else "EMA50<EMA200")

    # RSI зоны — немного ослабили, чтобы чаще были сигналы
    direction = None
    if trend_up:
        if 42 <= rsi_v <= 68:
            score += 35
            direction = "CALL"
            reasons.append("RSI ok for UP")
    else:
        if 32 <= rsi_v <= 58:
            score += 35
            direction = "PUT"
            reasons.append("RSI ok for DOWN")

    if direction is None:
        return None

    # бонус за волатильность
    if thr > 0:
        vol_bonus = int(min(18, (atrp / thr) * 6))
        score += vol_bonus
        reasons.append(f"ATR {atrp:.3f}% (thr {thr:.3f}%)")

    probability = max(55, min(92, int(score)))
    entry = now_tz()
    exit_ = entry + timedelta(minutes=EXPIRY_MINUTES)
    sid = f"{entry.strftime('%Y%m%d%H%M%S')}_{symbol.replace('/','')}"
    return Signal(
        symbol=symbol,
        direction=direction,
        probability=probability,
        price=close,
        rsi14=rsi_v,
        ema50=ema50_v,
        ema200=ema200_v,
        atr14_pct=float(atrp),
        entry_time=entry,
        exit_time=exit_,
        reason=" | ".join(reasons),
        id=sid,
    )

def pick_top_signals(symbols: List[str]) -> List[Signal]:
    signals: List[Signal] = []
    for s in symbols:
        try:
            sig = compute_signal(s)
            if sig:
                signals.append(sig)
        except TwelveLimitError:
            raise
        except Exception as e:
            log.warning("Signal error for %s: %s", s, e)

    signals.sort(key=lambda x: x.probability, reverse=True)
    if SEND_MODE == "BEST":
        return signals[:1]
    return signals[:TOP_N]

# =========================
# Telegram Messages
# =========================
def signal_message(sig: Signal) -> str:
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
        f"🆔 id: `{sig.id}`"
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
# Jobs
# =========================
async def job_expiry_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data or {}
    signal_id = data.get("signal_id")
    symbol = data.get("symbol")
    if not signal_id or not symbol:
        return
    await post_to_channel(
        context,
        f"⏱ Экспирация прошла по *{symbol}*.\n"
        f"Отметь результат кнопкой *WIN/LOSS* под сигналом.\n"
        f"🆔 id: `{signal_id}`",
        reply_markup=None
    )

async def job_send_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    global LAST_NO_SIGNAL_NOTICE, LAST_API_LIMIT_NOTICE

    now = now_tz()

    # антиспам "рынок слабый" (не чаще чем раз в 20 мин)
    def can_no_signal_notice() -> bool:
        return (LAST_NO_SIGNAL_NOTICE is None) or ((now - LAST_NO_SIGNAL_NOTICE).total_seconds() > 20 * 60)

    # антиспам "лимит API" (не чаще чем раз в 30 мин)
    def can_api_notice() -> bool:
        return (LAST_API_LIMIT_NOTICE is None) or ((now - LAST_API_LIMIT_NOTICE).total_seconds() > 30 * 60)

    # фильтр кулдауна по символу: если недавно отправляли — не берем его
    symbols = []
    for s in SYMBOLS:
        last = LAST_SENT_BY_SYMBOL.get(s)
        if last and (now - last).total_seconds() < COOLDOWN_MINUTES * 60:
            continue
        symbols.append(s)

    if not symbols:
        # нечего сканировать из-за кулдауна
        return

    try:
        top = pick_top_signals(symbols)
    except TwelveLimitError:
        if can_api_notice():
            LAST_API_LIMIT_NOTICE = now
            await post_to_channel(
                context,
                "⚠️ *Данные временно недоступны (лимит API).* \n"
                "Я не молчу — просто TwelveData ограничил запросы.\n"
                "✅ Решение: оставь *2 пары* и поставь интервал *10 минут*.\n"
                "Например:\n"
                "`SYMBOLS=EUR/USD,USD/JPY`\n"
                "`SIGNAL_INTERVAL_SECONDS=600`"
            )
        return

    # если сигналов нет — не молчим, но и не спамим
    if not top:
        if can_no_signal_notice():
            LAST_NO_SIGNAL_NOTICE = now
            await post_to_channel(
                context,
                "📉 *Рынок слабый сейчас* — сильных совпадений нет.\n"
                "Продолжаю анализ…"
            )
        return

    # отправляем сигналы
    sent_any = False
    for sig in top:
        if sig.probability < MIN_PROBABILITY:
            continue

        STATS["signals"] += 1
        STATS["last_signal"] = {"id": sig.id, "symbol": sig.symbol, "ts": fmt_dt(sig.entry_time)}
        LAST_SENT_BY_SYMBOL[sig.symbol] = now

        await post_to_channel(context, signal_message(sig), reply_markup=winloss_keyboard(sig.id))
        sent_any = True

        # авто-напоминание после экспирации
        if context.job_queue:
            context.job_queue.run_once(
                job_expiry_reminder,
                when=EXPIRY_MINUTES * 60,
                data={"signal_id": sig.id, "symbol": sig.symbol},
                name=f"expiry_{sig.id}",
            )

    if not sent_any:
        # если топ нашли, но ниже порога — мягко скажем (тоже с антиспамом)
        if can_no_signal_notice():
            LAST_NO_SIGNAL_NOTICE = now
            await post_to_channel(
                context,
                "📉 Есть движения, но *качество ниже порога* — сигналы не отправляю.\n"
                "Продолжаю анализ…"
            )

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
    )
    await post_to_channel(context, txt)

# =========================
# Handlers
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ Бот активен.\n"
        f"Канал: {CHANNEL_NAME}\n"
        f"Таймзона: {TIMEZONE_NAME}\n"
        f"Пары: {', '.join(SYMBOLS)}\n"
        f"TOP_N={TOP_N}, SEND_MODE={SEND_MODE}\n"
        f"ADAPTIVE_FILTERS={'ON' if ADAPTIVE_FILTERS else 'OFF'}\n\n"
        "Команды (владелец):\n"
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
        f"Пары: {', '.join(SYMBOLS)}\n"
        f"TOP_N={TOP_N}, SEND_MODE={SEND_MODE}\n"
        f"ADAPTIVE_FILTERS={'ON' if ADAPTIVE_FILTERS else 'OFF'}\n"
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

    parts = (q.data or "").split("|")
    if len(parts) != 3 or parts[0] != "wl":
        return

    action, signal_id = parts[1], parts[2]
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
    app.add_handler(CommandHandler("report_now", report_now))
    app.add_handler(CallbackQueryHandler(on_button))

    if app.job_queue is None:
        raise RuntimeError("JobQueue не активен. Установи python-telegram-bot[job-queue]==22.5")

    # Сигналы (TOP_N)
    app.job_queue.run_repeating(job_send_signals, interval=SIGNAL_INTERVAL_SECONDS, first=10, name="signals")

    # Пульс
    app.job_queue.run_repeating(job_pulse, interval=PULSE_INTERVAL_SECONDS, first=60, name="pulse")

    # Ежедневный отчет (по желанию)
    report_t = time(hour=REPORT_HOUR, minute=REPORT_MINUTE, tzinfo=TZ)
    app.job_queue.run_daily(job_daily_report, time=report_t, name="daily_report")

    log.info(
        "%s | started | TZ=%s | symbols=%s | interval=%ss | TOP_N=%d | SEND_MODE=%s | adaptive=%s",
        CHANNEL_NAME, TIMEZONE_NAME, ",".join(SYMBOLS), SIGNAL_INTERVAL_SECONDS, TOP_N, SEND_MODE, ADAPTIVE_FILTERS
    )

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
