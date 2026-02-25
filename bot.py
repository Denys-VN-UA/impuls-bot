# ==========================================
# IMPULS PRO v6.0  (REAL MARKET ENGINE)
# TwelveData Forex + Adaptive Logic
# ==========================================

import os
import requests
import pandas as pd
import datetime
import pytz
from telegram import Bot
from telegram.ext import Updater, CommandHandler

# ======================
# ENV
# ======================

TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHANNEL_ID")
TWELVE_KEY = os.getenv("TWELVE_API_KEY")

TIMEZONE = os.getenv("TIMEZONE", "Europe/Berlin")
SYMBOLS = os.getenv("SYMBOLS", "EUR/USD,GBP/USD,USD/JPY,USD/CHF,EUR/JPY").split(",")

MIN_PROBABILITY = int(os.getenv("MIN_PROBABILITY", 58))
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", 600))

TRADE_START_HOUR = int(os.getenv("TRADE_START_HOUR", 10))
TRADE_END_HOUR = int(os.getenv("TRADE_END_HOUR", 20))

bot = Bot(token=TOKEN)


# ======================
# TIME CHECK
# ======================

def is_trading_time():
    tz = pytz.timezone(TIMEZONE)
    now = datetime.datetime.now(tz)
    return TRADE_START_HOUR <= now.hour < TRADE_END_HOUR


# ======================
# MARKET DATA
# ======================

def get_data(symbol):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1min",
        "outputsize": 100,
        "apikey": TWELVE_KEY
    }
    r = requests.get(url, params=params).json()

    if "values" not in r:
        return None

    df = pd.DataFrame(r["values"])
    df = df.astype(float)
    df = df.sort_index(ascending=True)
    return df


# ======================
# INDICATORS
# ======================

def calculate_indicators(df):
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()

    return df


# ======================
# SIGNAL LOGIC
# ======================

def analyze_symbol(symbol):
    df = get_data(symbol)
    if df is None:
        return None

    df = calculate_indicators(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    probability = 50
    direction = None

    # Trend EMA
    if last["ema9"] > last["ema21"]:
        probability += 10
        direction = "⬆️ ВВЕРХ"
    elif last["ema9"] < last["ema21"]:
        probability += 10
        direction = "⬇️ ВНИЗ"

    # RSI
    if last["rsi"] < 30:
        probability += 10
        direction = "⬆️ ВВЕРХ"
    elif last["rsi"] > 70:
        probability += 10
        direction = "⬇️ ВНИЗ"

    # Momentum
    if last["close"] > prev["close"]:
        probability += 5
    else:
        probability += 5

    # Volatility adaptive (max +4)
    if last["atr"] > df["atr"].mean():
        probability += 4

    if probability >= MIN_PROBABILITY and direction:
        expiry = 3 if last["atr"] > df["atr"].mean() else 5
        return symbol, direction, probability, expiry

    return None


# ======================
# TELEGRAM
# ======================

def send_signal(context):
    if not is_trading_time():
        return

    best_signal = None

    for symbol in SYMBOLS:
        result = analyze_symbol(symbol)
        if result:
            if not best_signal or result[2] > best_signal[2]:
                best_signal = result

    if best_signal:
        symbol, direction, prob, expiry = best_signal
        message = (
            f"<b>📊 СИГНАЛ {symbol}</b>\n\n"
            f"<b>🎯 Направление:</b> {direction}\n"
            f"<b>🔥 Вероятность:</b> {prob}%\n"
            f"<b>⏱ Экспирация:</b> {expiry} мин\n"
        )
        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="HTML")


def start(update, context):
    update.message.reply_text("🚀 IMPULS PRO v6.0 запущен")


# ======================
# MAIN
# ======================

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))

    job_queue = updater.job_queue
    job_queue.run_repeating(send_signal, interval=SIGNAL_INTERVAL_SECONDS, first=10)

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
