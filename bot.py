# IMPULS ⚡ FINAL v4.1
# Hybrid 3/5 min • TwelveData • Auto expiry HTML report
# python-telegram-bot[job-queue]==22.5

import os
import logging
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes


# ================= ENV =================

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "")
CHANNEL_ID = os.getenv("CHANNEL_ID", "")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

TIMEZONE = os.getenv("TIMEZONE", "Europe/Kyiv")
TZ = ZoneInfo(TIMEZONE)

SYMBOLS = os.getenv(
    "SYMBOLS",
    "EUR/USD,GBP/USD,USD/JPY,USD/CHF,EUR/JPY"
).split(",")

SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "600"))
TOP_N = int(os.getenv("TOP_N", "1"))
MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", "55"))

ATR_THRESHOLD = float(os.getenv("ATR_THRESHOLD", "0.010"))
HYBRID_MODE = os.getenv("HYBRID_MODE", "1") == "1"
TREND_FILTER = os.getenv("TREND_FILTER", "1") == "1"

TRADE_START = "10:00"
TRADE_END = "20:00"

TD_BASE = "https://api.twelvedata.com"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("IMPULS")


# ================= UTILS =================

def now():
    return datetime.now(TZ)

def is_trading_time():
    n = now()
    if n.weekday() >= 5:
        return False
    start = time(10, 0, tzinfo=TZ)
    end = time(20, 0, tzinfo=TZ)
    return start <= n.timetz() <= end

def td_time_series(symbol, interval="1min", size=250):
    url = f"{TD_BASE}/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": size,
        "apikey": TWELVE_API_KEY,
        "format": "JSON"
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    if "values" not in data:
        raise RuntimeError("TwelveData error")
    df = pd.DataFrame(data["values"])
    df = df.iloc[::-1]
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def td_price(symbol):
    url = f"{TD_BASE}/price"
    params = {"symbol": symbol, "apikey": TWELVE_API_KEY}
    r = requests.get(url, params=params, timeout=10)
    return float(r.json()["price"])

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100/(1+rs))

def atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high-low).abs(),
         (high-prev_close).abs(),
         (low-prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def atr_percent(df):
    a = atr(df).iloc[-1]
    c = df["close"].iloc[-1]
    return float((a/c)*100)


# ================= SIGNAL =================

@dataclass
class Signal:
    symbol: str
    direction: str
    probability: int
    expiry: int
    entry_price: float
    entry_time: datetime


def compute_signal(symbol):

    df = td_time_series(symbol, "1min", 250)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi"] = rsi(df["close"], 14)

    close = float(df["close"].iloc[-1])
    ema50_v = float(df["ema50"].iloc[-1])
    ema200_v = float(df["ema200"].iloc[-1])
    rsi_v = float(df["rsi"].iloc[-1])
    atrp = atr_percent(df)

    if atrp < ATR_THRESHOLD:
        return None

    direction = None
    score = 0

    if ema50_v > ema200_v:
        if 45 <= rsi_v <= 65:
            direction = "CALL"
            score = 70
    elif ema50_v < ema200_v:
        if 35 <= rsi_v <= 55:
            direction = "PUT"
            score = 70

    if not direction:
        return None

    # HYBRID логика
    expiry = 3
    if HYBRID_MODE:
        if atrp > ATR_THRESHOLD * 1.5:
            expiry = 5

    probability = min(90, score + int(atrp*3))

    if probability < MIN_PROBABILITY:
        return None

    return Signal(
        symbol=symbol,
        direction=direction,
        probability=probability,
        expiry=expiry,
        entry_price=close,
        entry_time=now()
    )


# ================= TELEGRAM =================

def direction_label(d):
    return "⬆️ ВВЕРХ" if d=="CALL" else "⬇️ ВНИЗ"

def keyboard(sig_id):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ WIN", callback_data=f"w|{sig_id}"),
            InlineKeyboardButton("❌ LOSS", callback_data=f"l|{sig_id}")
        ]
    ])

async def send_signal(context, sig: Signal):

    exit_time = sig.entry_time + timedelta(minutes=sig.expiry)

    text = (
        f"📊 <b>СИГНАЛ {sig.symbol}</b>\n"
        f"🎯 Направление: <b>{direction_label(sig.direction)}</b>\n"
        f"🔥 Вероятность: <b>{sig.probability}%</b>\n"
        f"⌛️ Экспирация: <b>{sig.expiry} мин</b>\n\n"
        f"⏱ Вход: <b>{sig.entry_time.strftime('%H:%M:%S')}</b>\n"
        f"🏁 Выход: <b>{exit_time.strftime('%H:%M:%S')}</b>\n"
        f"🌍 {TIMEZONE}"
    )

    await context.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        parse_mode="HTML",
        reply_markup=keyboard(sig.entry_time.strftime("%H%M%S"))
    )

    # запуск авто отчета
    context.job_queue.run_once(
        expiry_report,
        when=sig.expiry*60 + 2,
        data=sig
    )


async def expiry_report(context: ContextTypes.DEFAULT_TYPE):
    sig: Signal = context.job.data

    try:
        last_price = td_price(sig.symbol)
    except:
        return

    move = "⬆️ ВВЕРХ" if last_price > sig.entry_price else "⬇️ ВНИЗ"
    win = (last_price > sig.entry_price) if sig.direction=="CALL" else (last_price < sig.entry_price)

    result = "✅ WIN" if win else "❌ LOSS"

    text = (
        f"⏱ <b>Экспирация {sig.expiry} мин</b> по <b>{sig.symbol}</b>\n"
        f"📈 <b>График пошёл:</b> {move}\n"
        f"✅ <b>По котировкам:</b> {result}"
    )

    await context.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        parse_mode="HTML"
    )


# ================= JOB =================

async def job_scan(context: ContextTypes.DEFAULT_TYPE):

    if not is_trading_time():
        return

    signals = []
    for s in SYMBOLS:
        try:
            sig = compute_signal(s.strip())
            if sig:
                signals.append(sig)
        except:
            continue

    signals.sort(key=lambda x: x.probability, reverse=True)

    for sig in signals[:TOP_N]:
        await send_signal(context, sig)


# ================= MAIN =================

def main():

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.job_queue.run_repeating(
        job_scan,
        interval=SIGNAL_INTERVAL_SECONDS,
        first=10
    )

    app.run_polling()


if __name__ == "__main__":
    main()
