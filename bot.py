# bot.py
# IMPULS ⚡ V3 HYBRID QUALITY — TwelveData
# python-telegram-bot[job-queue]==22.5

import os, logging, requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# =========================
# ENV
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "").strip()
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
TIMEZONE = os.getenv("TIMEZONE", "Europe/Kyiv")
TZ = ZoneInfo(TIMEZONE)

SYMBOLS = [s.strip() for s in os.getenv(
    "SYMBOLS",
    "EUR/USD,GBP/USD,USD/JPY,USD/CHF,EUR/JPY,GBP/JPY"
).split(",") if s.strip()]

SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "600"))
TOP_N = int(os.getenv("TOP_N", "2"))
MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", "62"))

BASE_ATR = float(os.getenv("ATR_THRESHOLD", "0.010"))
ADAPTIVE_FILTERS = os.getenv("ADAPTIVE_FILTERS", "1") == "1"

TREND_FILTER = True
TREND_TF = "15min"

TRADE_START = time(10, 0, tzinfo=TZ)
TRADE_END = time(20, 0, tzinfo=TZ)

# =========================
# LOG
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("IMPULS")

# =========================
# STATS
# =========================
STATS = {
    "signals": 0,
    "win": 0,
    "loss": 0,
    "cooldown": {}
}

# =========================
# UTILS
# =========================
def now():
    return datetime.now(TZ)

def trading_time():
    d = now()
    return d.weekday() < 5 and TRADE_START <= d.timetz() <= TRADE_END

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rsi(c, p=14):
    d = c.diff()
    u = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = u.ewm(alpha=1/p).mean() / dn.ewm(alpha=1/p).mean()
    return 100 - (100/(1+rs))

def atr_pct(df, p=14):
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p).mean().iloc[-1]
    return (atr / df.close.iloc[-1]) * 100

def td_candles(symbol, tf, n):
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": symbol,
        "interval": tf,
        "outputsize": n,
        "apikey": TWELVE_API_KEY,
        "format": "JSON"
    }, timeout=15)
    data = r.json()
    if "values" not in data:
        raise RuntimeError("API limit")
    df = pd.DataFrame(data["values"])[::-1]
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c])
    return df

def td_price(symbol):
    r = requests.get("https://api.twelvedata.com/price", params={
        "symbol": symbol,
        "apikey": TWELVE_API_KEY
    }, timeout=10).json()
    return float(r["price"])

# =========================
# SIGNAL
# =========================
@dataclass
class Signal:
    symbol: str
    direction: str
    prob: int
    entry: float
    expiry: int
    entry_time: datetime

def compute(symbol) -> Optional[Signal]:
    df = td_candles(symbol, "1min", 200)
    df["ema50"] = ema(df.close, 50)
    df["ema200"] = ema(df.close, 200)
    df["rsi"] = rsi(df.close)

    atr = atr_pct(df)
    thr = BASE_ATR
    if ADAPTIVE_FILTERS:
        thr = max(BASE_ATR, df.close.pct_change().abs().rolling(60).mean().iloc[-1]*100)

    if atr < thr:
        return None

    up = df.ema50.iloc[-1] > df.ema200.iloc[-1]
    down = df.ema50.iloc[-1] < df.ema200.iloc[-1]
    r = df.rsi.iloc[-1]

    direction = None
    score = 0

    if up and 45 <= r <= 65:
        direction = "CALL"
        score = 65
    elif down and 35 <= r <= 55:
        direction = "PUT"
        score = 65
    else:
        return None

    # старший тренд
    if TREND_FILTER:
        h = td_candles(symbol, TREND_TF, 120)
        h["ema50"] = ema(h.close, 50)
        h["ema200"] = ema(h.close, 200)
        if direction == "CALL" and h.ema50.iloc[-1] < h.ema200.iloc[-1]:
            return None
        if direction == "PUT" and h.ema50.iloc[-1] > h.ema200.iloc[-1]:
            return None
        score += 10

    prob = min(92, score + int(atr/thr*5))
    if prob < MIN_PROBABILITY:
        return None

    expiry = 5 if atr > thr*1.5 else 3

    return Signal(symbol, direction, prob, df.close.iloc[-1], expiry, now())

# =========================
# TELEGRAM
# =========================
def msg(sig: Signal):
    arrow = "⬆️" if sig.direction=="CALL" else "⬇️"
    return (
        f"📊 *{sig.symbol}*\n"
        f"{arrow} *{sig.direction}*\n"
        f"🔥 {sig.prob}%\n\n"
        f"💰 `{sig.entry:.5f}`\n"
        f"⏱ Экспирация: *{sig.expiry} мин*"
    )

async def post(ctx, text, kb=None):
    await ctx.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb
    )

# =========================
# JOB
# =========================
async def scan(ctx: ContextTypes.DEFAULT_TYPE):
    if not trading_time():
        return

    found = []
    for s in SYMBOLS:
        try:
            sig = compute(s)
            if sig:
                found.append(sig)
        except:
            continue

    if not found:
        await post(ctx, "📉 Рынок слабый — жду качество")
        return

    found.sort(key=lambda x: x.prob, reverse=True)
    for sig in found[:TOP_N]:
        STATS["signals"] += 1
        await post(ctx, msg(sig))

# =========================
# MAIN
# =========================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.job_queue.run_repeating(scan, SIGNAL_INTERVAL_SECONDS, first=10)
    app.run_polling()

if __name__ == "__main__":
    main()
