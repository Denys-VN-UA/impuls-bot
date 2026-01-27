import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ===================== ЛОГИ =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("impuls")

# ===================== НАСТРОЙКИ ИЗ RAILWAY VARIABLES =====================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "").strip()

CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
TIMEZONE = os.getenv("TIMEZONE", "Europe/Kyiv")

# если хочешь менять без кода — добавляй в Variables
SIGNAL_INTERVAL_SECONDS = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "180"))  # 3 мин
ENTRY_DELAY_SECONDS = int(os.getenv("ENTRY_DELAY_SECONDS", "30"))
EXPIRATION_MINUTES = int(os.getenv("EXPIRATION_MINUTES", "3"))

# режим отправки:
# ALL = отправлять каждый подходящий сигнал
# BEST = слать только лучший
SEND_MODE = os.getenv("SEND_MODE", "ALL").strip().upper()

# строгость (Pocket Option 1m лучше ниже)
MIN_PROB_TO_SEND = int(os.getenv("MIN_PROB_TO_SEND", "65"))

# анти-спам по одной паре
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "8"))

# индикаторы
RSI_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14

# сколько свечей брать
OUTPUTSIZE_1M = int(os.getenv("OUTPUTSIZE_1M", "260"))
OUTPUTSIZE_5M = int(os.getenv("OUTPUTSIZE_5M", "260"))

# --- Pocket Option (1m): ATR пороги в % (КЛЮЧЕВО, иначе сигналов нет) ---
ATR_THRESHOLDS = {
    "EUR/USD": float(os.getenv("ATR_EURUSD", "0.006")),
    "GBP/USD": float(os.getenv("ATR_GBPUSD", "0.007")),
    "USD/JPY": float(os.getenv("ATR_USDJPY", "0.006")),
    "AUD/USD": float(os.getenv("ATR_AUDUSD", "0.005")),
    "USD/CAD": float(os.getenv("ATR_USDCAD", "0.005")),
    "USD/CHF": float(os.getenv("ATR_USDCHF", "0.005")),
    "NZD/USD": float(os.getenv("ATR_NZDUSD", "0.005")),
}
DEFAULT_ATR_PCT = float(os.getenv("ATR_DEFAULT", "0.005"))

# хочешь ещё чаще — поставь меньше
GLOBAL_ATR_MULT = float(os.getenv("GLOBAL_ATR_MULT", "1.0"))  # 1.0 = как есть, 0.8 = мягче

# подтверждение 5m (жрёт API). По умолчанию выключено.
MTF_CONFIRM = os.getenv("MTF_CONFIRM", "0").strip() == "1"

PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"
]

# ===================== ПРОВЕРКИ =====================
def require_env():
    if not BOT_TOKEN:
        raise RuntimeError("❌ BOT_TOKEN пустой. Railway → Variables → BOT_TOKEN")
    if not TWELVE_API_KEY:
        raise RuntimeError("❌ TWELVE_API_KEY пустой. Railway → Variables → TWELVE_API_KEY")
    if CHANNEL_ID == 0:
        raise RuntimeError("❌ CHANNEL_ID пустой. Railway → Variables → CHANNEL_ID")
    if OWNER_ID == 0:
        raise RuntimeError("❌ OWNER_ID пустой. Railway → Variables → OWNER_ID")

# ===================== СТАТЫ / ПАМЯТЬ =====================
STATS = {"win": 0, "loss": 0}
LAST_SENT = {}  # pair -> datetime

# ===================== КНОПКИ =====================
def winloss_keyboard():
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ WIN", callback_data="WIN"),
        InlineKeyboardButton("❌ LOSS", callback_data="LOSS"),
    ]])

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "WIN":
        STATS["win"] += 1
    elif query.data == "LOSS":
        STATS["loss"] += 1

    total = STATS["win"] + STATS["loss"]
    winrate = round((STATS["win"] / total) * 100, 1) if total else 0.0

    await query.message.reply_text(
        f"📊 Статистика\nWIN: {STATS['win']}\nLOSS: {STATS['loss']}\nWinrate: {winrate}%"
    )

# ===================== INDICATORS =====================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    x = df.copy()
    if "high" in x.columns and "low" in x.columns:
        prev_close = x["close"].shift(1)
        tr = pd.concat([
            (x["high"] - x["low"]),
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
    else:
        tr = x["close"].diff().abs()
    return tr.rolling(period).mean()

# ===================== DATA (TWELVE) =====================
def get_market_data(pair: str, interval: str, outputsize: int):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
    except Exception as e:
        return None, f"Request error: {e}"

    if data.get("status") == "error":
        return None, data.get("message", "API error")

    values = data.get("values")
    if not values:
        return None, "No candle data"

    df = pd.DataFrame(values)
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col].astype(float)

    df = df.sort_values("datetime")
    return df, None

# ===================== SIGNAL LOGIC (ослаблено) =====================
def build_signal_1m(df: pd.DataFrame):
    x = df.copy()
    x["ema50"] = calculate_ema(x["close"], EMA_FAST)
    x["ema200"] = calculate_ema(x["close"], EMA_SLOW)
    x["rsi"] = calculate_rsi(x["close"], RSI_PERIOD)

    last = x.iloc[-1]
    price = float(last["close"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi = float(last["rsi"]) if pd.notna(last["rsi"]) else None

    snap = {"price": price, "ema50": ema50, "ema200": ema200, "rsi": rsi}

    if rsi is None:
        return None, 0, "Not enough RSI", snap

    # Ослабляем: флэт уже 47-53 (а не 45-55)
    if 47 <= rsi <= 53:
        return None, 0, "Flat RSI (47–53)", snap

    # Экстремумы ослабляем: только если совсем жёстко
    if rsi >= 78 or rsi <= 22:
        return None, 0, "RSI hard extreme", snap

    trend_up = ema50 > ema200
    trend_down = ema50 < ema200
    if not (trend_up or trend_down):
        return None, 0, "No EMA trend", snap

    # тренд-сила (мягче чем раньше)
    trend_strength = abs(ema50 - ema200) / price * 100
    ts = min(1.0, trend_strength / 0.12)  # раньше было 0.20 (жёстко)

    ideal = 55 if trend_up else 45
    rsi_dist = abs(rsi - ideal)
    rs = max(0.0, 1.0 - (rsi_dist / 22.0))

    price_ok = (price > ema50) if trend_up else (price < ema50)
    ps = 1.0 if price_ok else 0.2  # раньше было 0/1

    score01 = 0.45 * ts + 0.35 * rs + 0.20 * ps

    # порог сигнала мягче
    if score01 < 0.52:
        return None, 0, f"Weak score ({score01:.2f})", snap

    probability = int(round(55 + score01 * 40))  # чаще 65-90
    probability = max(55, min(90, probability))

    direction = "CALL" if trend_up else "PUT"
    reason = f"RSI={rsi:.1f}; EMA50 {'>' if trend_up else '<'} EMA200; price_ok={price_ok}"
    return direction, probability, reason, snap

def direction_confirm_5m(df: pd.DataFrame):
    x = df.copy()
    x["ema50"] = calculate_ema(x["close"], EMA_FAST)
    x["ema200"] = calculate_ema(x["close"], EMA_SLOW)
    last = x.iloc[-1]
    if float(last["ema50"]) > float(last["ema200"]):
        return "CALL"
    if float(last["ema50"]) < float(last["ema200"]):
        return "PUT"
    return None

def format_signal_text(pair: str, direction: str, probability: int, snap: dict, atr_pct: float):
    entry = datetime.now() + timedelta(seconds=ENTRY_DELAY_SECONDS)
    exit_ = entry + timedelta(minutes=EXPIRATION_MINUTES)
    if direction == "CALL":
    arrow = "📈⬆️"
    dir_text = "CALL (вверх)"
    trend_text = "📈 ТРЕНД ВВЕРХ"
else:
    arrow = "📉⬇️"
    dir_text = "PUT (вниз)"
    trend_text = "📉 ТРЕНД ВНИЗ"

    return (
        f"📊 СИГНАЛ {pair}\n"
        f"{arrow} {direction}\n"
        f"🔥 Вероятность: {probability}%\n"
        f"📐 ATR(14): {atr_pct:.3f}%\n\n"
        f"💰 Цена: {snap['price']:.5f}\n"
        f"📉 RSI(14): {snap['rsi']:.1f}\n"
        f"📈 EMA50: {snap['ema50']:.5f}\n"
        f"📈 EMA200: {snap['ema200']:.5f}\n\n"
        f"⏰ Вход: {entry.strftime('%H:%M:%S')}\n"
        f"⌛ Экспирация: {EXPIRATION_MINUTES} мин\n"
        f"🏁 Выход: {exit_.strftime('%H:%M:%S')}"
    )

# ===================== COMMANDS =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Бот активен.\n"
        "/test — тест в канал\n"
        "/stats — статистика\n"
        "Сигналы идут в канал автоматически."
    )

async def test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Тест: отправляю сообщение в канал…")
    await context.bot.send_message(chat_id=CHANNEL_ID, text="✅ ТЕСТ: бот может писать в канал (OK)")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = STATS["win"] + STATS["loss"]
    winrate = round((STATS["win"] / total) * 100, 1) if total else 0.0
    await update.message.reply_text(
        f"📊 Статистика\nWIN: {STATS['win']}\nLOSS: {STATS['loss']}\nWinrate: {winrate}%"
    )

# ===================== JOB =====================
async def job_scan(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now()

    sent_any = False

    for pair in PAIRS:
        # анти-спам
        last_time = LAST_SENT.get(pair)
        if last_time and (now - last_time).total_seconds() < COOLDOWN_MINUTES * 60:
            continue

        df1, err1 = get_market_data(pair, interval="1min", outputsize=OUTPUTSIZE_1M)
        if err1 or df1 is None or len(df1) < 220:
            logger.info("skip %s: %s", pair, err1)
            continue

        atr_series = calculate_atr(df1, ATR_PERIOD)
        atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
        price = float(df1["close"].iloc[-1])
        atr_pct = (atr / price) * 100 if price else 0.0

        threshold = ATR_THRESHOLDS.get(pair, DEFAULT_ATR_PCT) * GLOBAL_ATR_MULT
        if atr_pct < threshold:
            logger.info("low vol %s atr%%=%.3f (thr=%.3f)", pair, atr_pct, threshold)
            continue

        direction1, prob1, reason1, snap1 = build_signal_1m(df1)
        if direction1 is None:
            logger.info("no signal %s: %s", pair, reason1)
            continue

        if MTF_CONFIRM:
            df5, err5 = get_market_data(pair, interval="5min", outputsize=OUTPUTSIZE_5M)
            if err5 or df5 is None or len(df5) < 220:
                logger.info("skip 5m %s: %s", pair, err5)
                continue
            direction5 = direction_confirm_5m(df5)
            if direction5 != direction1:
                logger.info("mtf reject %s 1m=%s 5m=%s", pair, direction1, direction5)
                continue
            prob1 = min(90, prob1 + 3)

        if prob1 < MIN_PROB_TO_SEND:
            logger.info("prob low %s prob=%s (<%s)", pair, prob1, MIN_PROB_TO_SEND)
            continue

        # режим ALL = шлём всё подходящее
        if SEND_MODE == "ALL":
            text = format_signal_text(pair, direction1, prob1, snap1, atr_pct)
            await context.bot.send_message(chat_id=CHANNEL_ID, text=text, reply_markup=winloss_keyboard())
            LAST_SENT[pair] = now
            sent_any = True
            logger.info("SENT %s %s prob=%s", pair, direction1, prob1)

        # режим BEST = выбираем лучший (если вдруг захочешь)
        else:
            # для BEST — просто держим один лучший
            context.chat_data.setdefault("best", None)
            best = context.chat_data["best"]
            cand = (prob1, pair, direction1, snap1, atr_pct)
            if best is None or cand[0] > best[0]:
                context.chat_data["best"] = cand

    if SEND_MODE != "ALL":
        best = context.chat_data.get("best")
        context.chat_data["best"] = None
        if not best:
            logger.info("no strong signals this cycle")
            return
        prob, pair, direction, snap, atr_pct = best
        text = format_signal_text(pair, direction, prob, snap, atr_pct)
        await context.bot.send_message(chat_id=CHANNEL_ID, text=text, reply_markup=winloss_keyboard())
        LAST_SENT[pair] = now
        sent_any = True
        logger.info("SENT BEST %s %s prob=%s", pair, direction, prob)

    if not sent_any:
        logger.info("cycle done: no sends")

# ===================== MAIN =====================
def main():
    require_env()
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("test", test))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CallbackQueryHandler(on_button))

    if app.job_queue is None:
        raise RuntimeError("JobQueue не активен. Нужен пакет: python-telegram-bot[job-queue]")

    app.job_queue.run_repeating(job_scan, interval=SIGNAL_INTERVAL_SECONDS, first=10)

    print("🚀 Bot started. Mode:", SEND_MODE, "| MTF_CONFIRM:", MTF_CONFIRM, "| TZ:", TIMEZONE)
    app.run_polling()

if __name__ == "__main__":
    main()
