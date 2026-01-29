# bot.py
# IMPULS ⚡ Alpha Vantage + TOP_N + ADAPTIVE_FILTERS + Auto-report after expiry
# python-telegram-bot[job-queue]==22.5

import os
import logging
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# =========================
# ENV НАСТРОЙКИ
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

# Alpha Vantage
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()

# Channel
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()   # -100... или @username

# Owner
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

CHANNEL_NAME = os.getenv("CHANNEL_NAME", "IMPULS ⚡").strip()

# TZ
TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Kyiv").strip()
TZ = ZoneInfo(TIMEZONE_NAME)

# Symbols (forex pairs)
# Формат: EUR/USD,USD/JPY,USD/CHF
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "USD/CHF,USD/JPY").split(",") if s.strip()]

# Jobs intervals
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "600"))  # 10 минут (важно для free API)
PULSE_INTERVAL_SECONDS = int(os.getenv("PULSE_INTERVAL_SECONDS", "600"))    # 10 минут

# Trading params
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "3"))
ENTRY_DELAY_SECONDS = int(os.getenv("ENTRY_DELAY_SECONDS", "0"))  # если хочешь вход не сразу — поставь 30

# Filters
ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.020"))  # в % (пример: 0.020%)
MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", "70"))   # минимум вероятности
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "10")) # анти-спам по паре

# Mode
SEND_MODE = os.getenv("SEND_MODE", "BEST").strip().upper()  # BEST | ALL
TOP_N = int(os.getenv("TOP_N", "1"))                        # 1..3

# Adaptive
ADAPTIVE_FILTERS = os.getenv("ADAPTIVE_FILTERS", "0").strip() in ("1", "true", "True", "YES", "yes")

# Alpha Vantage interval
# FX_INTRADAY supports: 1min, 5min, 15min, 30min, 60min
TF = os.getenv("TF", "5min").strip()
CANDLES = int(os.getenv("CANDLES", "250"))

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("impuls")

# =========================
# STATE (in-memory)
# =========================
STATS = {"signals": 0, "win": 0, "loss": 0, "pulse_on": True}

# last sent per pair (cooldown)
LAST_SENT: Dict[str, datetime] = {}

# Signals registry: signal_id -> dict(status/outcome/signal)
SIGNALS: Dict[str, dict] = {}

# =========================
# HELPERS
# =========================
def now_tz() -> datetime:
    return datetime.now(TZ)

def fmt_time(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%H:%M:%S")

def fmt_dt(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%d.%m.%Y %H:%M:%S")

def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

def require_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN пустой. Добавь BOT_TOKEN в Railway Variables.")
    if not ALPHAVANTAGE_API_KEY:
        raise RuntimeError("ALPHAVANTAGE_API_KEY пустой. Добавь ALPHAVANTAGE_API_KEY в Railway Variables.")
    if not CHANNEL_ID:
        raise RuntimeError("CHANNEL_ID пустой. Добавь CHANNEL_ID в Railway Variables.")
    if OWNER_ID == 0:
        log.warning("OWNER_ID = 0. WIN/LOSS и owner-команды будут недоступны.")

def direction_label(direction: str) -> str:
    return "⬆️ ВВЕРХ" if direction == "CALL" else "⬇️ ВНИЗ"

# =========================
# INDICATORS
# =========================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
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
    if pd.isna(a) or pd.isna(c) or c == 0:
        return 0.0
    return float((a / c) * 100.0)

# =========================
# ALPHA VANTAGE (FX_INTRADAY)
# =========================
AV_BASE = "https://www.alphavantage.co/query"

def av_fx_intraday(symbol: str, interval: str, outputsize: str = "compact") -> pd.DataFrame:
    """
    symbol format: 'USD/JPY'
    Alpha Vantage wants from_symbol, to_symbol.
    """
    if "/" not in symbol:
        raise RuntimeError(f"Bad symbol format: {symbol} (need like USD/JPY)")
    f, t = symbol.split("/", 1)

    params = {
        "function": "FX_INTRADAY",
        "from_symbol": f,
        "to_symbol": t,
        "interval": interval,
        "outputsize": outputsize,  # compact ~100, full ~10000 (but slower)
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    r = requests.get(AV_BASE, params=params, timeout=25)
    data = r.json()

    # API limit / notes
    note = data.get("Note") or data.get("Information")
    if note:
        raise RuntimeError("API_LIMIT")

    # key example: "Time Series FX (5min)"
    ts_key = None
    for k in data.keys():
        if "Time Series FX" in k:
            ts_key = k
            break
    if not ts_key or ts_key not in data:
        raise RuntimeError(f"No time series for {symbol}: {data}")

    ts = data[ts_key]  # dict of "YYYY-MM-DD HH:MM:SS": {1. open, 2. high, 3. low, 4. close}
    rows = []
    for dt_str, ohlc in ts.items():
        rows.append({
            "datetime": dt_str,
            "open": float(ohlc.get("1. open", np.nan)),
            "high": float(ohlc.get("2. high", np.nan)),
            "low": float(ohlc.get("3. low", np.nan)),
            "close": float(ohlc.get("4. close", np.nan)),
        })

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)

    # keep last CANDLES
    if len(df) > CANDLES:
        df = df.iloc[-CANDLES:].reset_index(drop=True)

    return df

# =========================
# SIGNAL LOGIC
# =========================
@dataclass
class Signal:
    symbol: str
    direction: str      # CALL/PUT
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
    df = av_fx_intraday(symbol, TF, outputsize="compact")

    if len(df) < 220:
        return None

    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)

    close = float(df["close"].iloc[-1])
    ema50_v = float(df["ema50"].iloc[-1])
    ema200_v = float(df["ema200"].iloc[-1])
    rsi_v = float(df["rsi14"].iloc[-1])
    atr_pct = atr_percent(df, 14)

    # --- Adaptive ATR threshold (optional) ---
    thr = ATR_THRESHOLD
    if ADAPTIVE_FILTERS:
        # берём медиану ATR% за последние 60 свечей и ставим порог ~70% от неё, но не ниже базового
        recent = []
        try:
            a_series = atr(df, 14)
            c_series = df["close"]
            atrp = (a_series / c_series) * 100.0
            recent = atrp.dropna().iloc[-60:].tolist()
        except Exception:
            recent = []

        if recent:
            med = float(np.median(recent))
            thr = max(ATR_THRESHOLD, med * 0.70)

    if atr_pct < thr:
        return None

    trend_up = ema50_v > ema200_v
    trend_down = ema50_v < ema200_v
    if not (trend_up or trend_down):
        return None

    score = 0
    reasons = []

    # Trend component
    score += 35
    reasons.append("EMA50 выше EMA200" if trend_up else "EMA50 ниже EMA200")

    # RSI “impulse zone” (чуть мягче, чтобы чаще давал сигналы)
    direction = None
    if trend_up:
        if 43 <= rsi_v <= 68:
            score += 35
            direction = "CALL"
            reasons.append("RSI подтверждает вверх")
        else:
            reasons.append("RSI не подтверждает вверх")
    else:
        if 32 <= rsi_v <= 57:
            score += 35
            direction = "PUT"
            reasons.append("RSI подтверждает вниз")
        else:
            reasons.append("RSI не подтверждает вниз")

    if direction is None:
        return None

    # Volatility bonus
    vol_bonus = min(20, int((atr_pct / max(thr, 0.0001)) * 6))
    score += vol_bonus
    reasons.append(f"ATR(14) {atr_pct:.3f}% (thr {thr:.3f}%)")

    probability = int(max(55, min(92, score)))

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
        reason=" | ".join(reasons),
    )

def pick_signals(symbols: List[str]) -> List[Signal]:
    out: List[Signal] = []
    for sym in symbols:
        try:
            sig = compute_signal(sym)
            if not sig:
                continue
            if sig.probability < MIN_PROBABILITY:
                continue
            out.append(sig)
        except RuntimeError as e:
            if str(e) == "API_LIMIT":
                raise
            log.warning("Signal error %s: %s", sym, e)
        except Exception as e:
            log.warning("Signal error %s: %s", sym, e)

    out.sort(key=lambda s: s.probability, reverse=True)
    return out

# =========================
# TELEGRAM MSGS
# =========================
def winloss_keyboard(signal_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ WIN", callback_data=f"wl|win|{signal_id}"),
            InlineKeyboardButton("❌ LOSS", callback_data=f"wl|loss|{signal_id}"),
        ]
    ])

def signal_text(sig: Signal) -> str:
    dir_txt = direction_label(sig.direction)
    arrow = "📈" if sig.direction == "CALL" else "📉"
    return (
        f"📊 *СИГНАЛ {sig.symbol}* {arrow}\n"
        f"📌 Направление: *{dir_txt}*\n"
        f"🔥 Вероятность: *{sig.probability}%*\n"
        f"⚡ ATR(14): `{sig.atr14_pct:.3f}%`\n\n"
        f"💰 Цена: `{sig.price:.5f}`\n"
        f"📉 RSI(14): `{sig.rsi14:.1f}`\n"
        f"📍 EMA50: `{sig.ema50:.5f}`\n"
        f"📍 EMA200: `{sig.ema200:.5f}`\n\n"
        f"⏱ Вход: *{fmt_time(sig.entry_time)}*\n"
        f"🏁 Выход: *{fmt_time(sig.exit_time)}*  (экспирация {EXPIRY_MINUTES} мин)\n"
        f"🌍 Таймзона: `{TIMEZONE_NAME}`"
    )

def expiry_report_text(sig: Signal, outcome: Optional[str]) -> str:
    # outcome: "win" | "loss" | None
    wr = (STATS["win"] / max(1, STATS["win"] + STATS["loss"])) * 100.0
    res_line = "⏳ Результат: *НЕ ОТМЕЧЕН*" if outcome is None else ("✅ Результат: *WIN*" if outcome == "win" else "❌ Результат: *LOSS*")
    return (
        f"📌 *ОТЧЁТ ПО СДЕЛКЕ*\n"
        f"📊 Пара: *{sig.symbol}*\n"
        f"📌 Направление: *{direction_label(sig.direction)}*\n"
        f"{res_line}\n\n"
        f"⏱ Вход: *{fmt_time(sig.entry_time)}*\n"
        f"🏁 Выход: *{fmt_time(sig.exit_time)}*\n\n"
        f"📈 Всего сигналов: *{STATS['signals']}*\n"
        f"✅ WIN: *{STATS['win']}*   ❌ LOSS: *{STATS['loss']}*\n"
        f"🎯 WinRate: *{wr:.1f}%*\n"
    )

async def post(context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None) -> None:
    await context.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )

# =========================
# AUTO REPORT JOB (after expiry)
# =========================
async def job_trade_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data or {}
    signal_id = data.get("signal_id")
    if not signal_id or signal_id not in SIGNALS:
        return

    rec = SIGNALS[signal_id]
    sig: Signal = rec["signal"]
    outcome = rec.get("outcome")  # None/win/loss

    # всегда отправляем авто-отчёт после экспирации
    text = expiry_report_text(sig, outcome)

    # если исход не отмечен — дать кнопки ещё раз
    kb = winloss_keyboard(signal_id) if outcome is None else None
    await post(context, text, reply_markup=kb)

# =========================
# JOB: SEND SIGNALS
# =========================
async def job_send_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    now = now_tz()

    # cooldown filter (по символу)
    symbols = []
    for s in SYMBOLS:
        last = LAST_SENT.get(s)
        if last and (now - last).total_seconds() < COOLDOWN_MINUTES * 60:
            continue
        symbols.append(s)

    if not symbols:
        return

    try:
        signals = pick_signals(symbols)
    except RuntimeError as e:
        if str(e) == "API_LIMIT":
            await post(context, "⚠️ Данные временно недоступны (лимит API). Уменьши пары или увеличь интервал.")
            return
        raise

    if not signals:
        await post(context, "🟡 Рынок слабый: подходящих сигналов сейчас нет (фильтры/волатильность).")
        return

    to_send: List[Signal]
    if SEND_MODE == "ALL":
        to_send = signals[:max(1, TOP_N)]
    else:
        # BEST (top N)
        to_send = signals[:max(1, TOP_N)]

    for sig in to_send:
        STATS["signals"] += 1

        signal_id = f"{sig.entry_time.strftime('%Y%m%d%H%M%S')}_{sig.symbol.replace('/','')}"
        SIGNALS[signal_id] = {"signal": sig, "outcome": None, "created": fmt_dt(now)}

        await post(context, signal_text(sig), reply_markup=winloss_keyboard(signal_id))
        LAST_SENT[sig.symbol] = now

        # --- AUTO REPORT after expiry (+5 sec safety) ---
        report_when = sig.exit_time + timedelta(seconds=5)
        context.job_queue.run_once(
            job_trade_report,
            when=report_when,
            data={"signal_id": signal_id},
            name=f"trade_report_{signal_id}",
        )

# =========================
# JOB: PULSE
# =========================
async def job_pulse(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not STATS.get("pulse_on", True):
        return
    await post(context, f"🕒 *{CHANNEL_NAME}*: бот жив, анализирую рынок…")

# =========================
# HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ Бот активен.\n\n"
        f"Канал: {CHANNEL_NAME}\n"
        f"Пары: {', '.join(SYMBOLS)}\n"
        f"TF: {TF}\n"
        f"TOP_N: {TOP_N} | SEND_MODE: {SEND_MODE}\n"
        f"ADAPTIVE_FILTERS: {'ON' if ADAPTIVE_FILTERS else 'OFF'}\n"
        f"Таймзона: {TIMEZONE_NAME}\n\n"
        "Команды (только владелец):\n"
        "/test /stats /pulse_on /pulse_off\n",
        disable_web_page_preview=True,
    )

async def test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    await post(context, "✅ *ТЕСТ*: бот может писать в канал (OK)")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_owner(update.effective_user.id):
        return
    wr = (STATS["win"] / max(1, STATS["win"] + STATS["loss"])) * 100.0
    await update.message.reply_text(
        f"📊 Статистика\n"
        f"Сигналов: {STATS['signals']}\n"
        f"WIN: {STATS['win']}\n"
        f"LOSS: {STATS['loss']}\n"
        f"WinRate: {wr:.1f}%\n"
        f"ADAPTIVE_FILTERS: {'ON' if ADAPTIVE_FILTERS else 'OFF'}\n"
        f"TOP_N: {TOP_N} | SEND_MODE: {SEND_MODE}"
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

    action, signal_id = parts[1], parts[2]
    if signal_id not in SIGNALS:
        await q.message.reply_text("⚠️ Этот сигнал уже не найден (перезапуск бота).")
        return

    rec = SIGNALS[signal_id]
    if rec.get("outcome") is not None:
        await q.message.reply_text("ℹ️ Уже отмечено ранее.")
        return

    if action == "win":
        STATS["win"] += 1
        rec["outcome"] = "win"
        await q.message.reply_text(f"✅ WIN отмечен (id={signal_id})")
    elif action == "loss":
        STATS["loss"] += 1
        rec["outcome"] = "loss"
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

    # Signals loop
    app.job_queue.run_repeating(job_send_signals, interval=SIGNAL_INTERVAL_SECONDS, first=10)

    # Pulse loop
    app.job_queue.run_repeating(job_pulse, interval=PULSE_INTERVAL_SECONDS, first=60)

    log.info("IMPULS started | symbols=%s | TF=%s | interval=%ss | TOP_N=%s | SEND_MODE=%s | adaptive=%s",
             SYMBOLS, TF, SIGNAL_INTERVAL_SECONDS, TOP_N, SEND_MODE, ADAPTIVE_FILTERS)

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
