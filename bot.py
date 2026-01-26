import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Optional, Dict, Any, Tuple

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ===================== НАСТРОЙКИ =====================

# Можно вставить прямо сюда, или задать через переменные окружения:
# BOT_TOKEN, TWELVE_API_KEY
import os

BOT_TOKEN = os.getenv("BOT_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

if not BOT_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN не задан в Railway → Variables")
if not TWELVE_API_KEY:
    raise RuntimeError("❌ TWELVE_API_KEY не задан в Railway → Variables")
if CHANNEL_ID == 0:
    raise RuntimeError("❌ CHANNEL_ID не задан в Railway → Variables")
if OWNER_ID == 0:
    raise RuntimeError("❌ OWNER_ID не задан в Railway → Variables")
# Название канала (для текста отчётов)
CHANNEL_NAME = "IMPULS"

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"]

SIGNAL_INTERVAL_SECONDS = 180   # 3 минуты
ENTRY_DELAY_SECONDS = 30        # вход через 30 сек
EXPIRATION_MINUTES = 3          # экспирация 3 минуты
EVAL_EXTRA_SECONDS = 10         # запас после выхода (чтобы свеча успела обновиться)

RSI_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
MIN_BARS_FOR_INDICATORS = 220

# Минимальная вероятность (строгость)
MIN_PROB_TO_SEND = 75

# анти-спам по одной паре
COOLDOWN_MINUTES = 15

# Пульс в канал (чтобы знать, что бот жив)
PULSE_ENABLED_DEFAULT = False

# -------- 1) ФИЛЬТР СЕССИЙ (локальное время твоего Mac) --------
SESSION_RULES = {
    "EUR/USD": [("10:00", "22:00")],
    "GBP/USD": [("10:00", "22:00")],
    "AUD/USD": [("02:00", "12:00")],
    "USD/JPY": [("02:00", "22:00")],
    "USD/CAD": [("14:00", "22:00")],
    "USD/CHF": [("10:00", "22:00")],
    "NZD/USD": [("02:00", "12:00")],
}

# -------- 2) ATR пороги по парам (в %) --------
ATR_THRESHOLDS = {
    "EUR/USD": 0.020,
    "GBP/USD": 0.022,
    "USD/JPY": 0.028,
    "AUD/USD": 0.020,
    "USD/CAD": 0.020,
    "USD/CHF": 0.020,
    "NZD/USD": 0.020,
}
DEFAULT_ATR_PCT = 0.020

# -------- 4) ПАУЗЫ ПОСЛЕ LOSS --------
LOSS_STREAK = 0
GLOBAL_PAUSE_UNTIL: Optional[datetime] = None
PAUSE_AFTER_1_LOSS_MIN = 5
PAUSE_AFTER_2_LOSS_MIN = 30
PAUSE_AFTER_3_LOSS_MIN = 180  # 3 часа

# -------- АВТООТЧЁТ --------
REPORT_HOUR = 22
REPORT_MINUTE = 0

# ===================== ЛОГИ =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("trade_bot")

# ===================== СТАТИСТИКА =====================
STATS = {"win": 0, "loss": 0}         # общая
DAY_STATS = {"win": 0, "loss": 0, "trades": 0}  # дневная
DAY_BEST: Dict[str, int] = {}         # pair -> max prob today

PULSE_ENABLED = PULSE_ENABLED_DEFAULT

LAST_SENT: Dict[str, datetime] = {}      # pair -> datetime (анти-спам)
LOWVOL_STATE: Dict[str, bool] = {}       # pair -> bool (для алерта "рынок ожил")

# Последняя отправленная сделка (для авто-оценки)
TRADES: Dict[str, Dict[str, Any]] = {}   # trade_id -> data

# Чтобы в выходные не спамить одинаковым статусом
LAST_WEEKEND_NOTICE_DATE: Optional[str] = None


# ===================== ВСПОМОГАТЕЛЬНОЕ =====================

def is_market_open_now() -> bool:
    """
    Для твоих пар (Forex): суббота/воскресенье рынок закрыт.
    """
    wd = datetime.now().weekday()  # 0=Mon ... 5=Sat 6=Sun
    return wd not in (5, 6)

def in_session(pair: str) -> bool:
    rules = SESSION_RULES.get(pair)
    if not rules:
        return True
    now = datetime.now().strftime("%H:%M")
    for start, end in rules:
        if start <= now <= end:
            return True
    return False

def winloss_keyboard(trade_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ WIN", callback_data=f"WIN|{trade_id}"),
        InlineKeyboardButton("❌ LOSS", callback_data=f"LOSS|{trade_id}"),
    ]])

def calc_winrate(win: int, loss: int) -> float:
    total = win + loss
    return round((win / total) * 100, 1) if total else 0.0


# ===================== ИНДИКАТОРЫ =====================

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
    """
    ATR по high/low/close если есть. Если нет — fallback по |diff(close)|.
    """
    x = df.copy()

    if "high" in x.columns and "low" in x.columns:
        x["high"] = x["high"].astype(float)
        x["low"] = x["low"].astype(float)
        prev_close = x["close"].shift(1)
        tr = pd.concat([
            (x["high"] - x["low"]),
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
    else:
        tr = x["close"].diff().abs()

    return tr.rolling(period).mean()


# ===================== TWELVE DATA =====================

def get_market_data(pair: str, interval: str = "1min", outputsize: int = 300) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not TWELVE_API_KEY or "PASTE_" in TWELVE_API_KEY:
        return None, "TwelveData API key is not set"

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


# ===================== АЛЕРТ ВОЛАТИЛЬНОСТИ =====================

async def send_volatility_alert(
    context: ContextTypes.DEFAULT_TYPE,
    pair: str,
    atr_pct: float,
    threshold: float
):
    await context.bot.send_message(
        chat_id=CHANNEL_ID,
        text=(
            f"🔥 Волатильность появилась: {pair}\n"
            f"📐 ATR(14): {atr_pct:.3f}% (порог {threshold:.3f}%)\n"
            f"🔍 Ищу лучший сигнал на {EXPIRATION_MINUTES} минуты…"
        )
    )


# ===================== ЛОГИКА СИГНАЛА =====================

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
        return None, 0, "Not enough RSI data", snap

    # Флэт и экстремумы — пропуск
    if 45 <= rsi <= 55:
        return None, 0, "Flat RSI (45–55)", snap
    if rsi >= 70 or rsi <= 30:
        return None, 0, "RSI extreme", snap

    trend_up = ema50 > ema200
    trend_down = ema50 < ema200
    if not (trend_up or trend_down):
        return None, 0, "No EMA trend", snap

    # сила тренда (EMA50-EMA200) в % от цены
    trend_strength = abs(ema50 - ema200) / price * 100
    ts = min(1.0, trend_strength / 0.20)  # 0..1

    ideal = 55 if trend_up else 45
    rsi_dist = abs(rsi - ideal)
    rs = max(0.0, 1.0 - (rsi_dist / 20.0))  # 0..1

    price_ok = (price > ema50) if trend_up else (price < ema50)
    ps = 1.0 if price_ok else 0.0

    score01 = 0.45 * ts + 0.35 * rs + 0.20 * ps

    if score01 < 0.62:
        return None, 0, f"Weak score ({score01:.2f})", snap

    probability = int(round(55 + score01 * 35))  # ~77..90
    probability = max(60, min(90, probability))

    direction = "CALL" if trend_up else "PUT"
    reason = f"RSI={rsi:.1f}; EMA50={'>' if trend_up else '<'}EMA200; Confirm={'yes' if price_ok else 'no'}"
    return direction, probability, reason, snap

def direction_confirm_5m(df: pd.DataFrame):
    x = df.copy()
    x["ema50"] = calculate_ema(x["close"], EMA_FAST)
    x["ema200"] = calculate_ema(x["close"], EMA_SLOW)
    x["rsi"] = calculate_rsi(x["close"], RSI_PERIOD)

    last = x.iloc[-1]
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi = float(last["rsi"]) if pd.notna(last["rsi"]) else None

    if rsi is None:
        return None
    if 45 <= rsi <= 55:
        return None
    if rsi >= 70 or rsi <= 30:
        return None

    if ema50 > ema200:
        return "CALL"
    if ema50 < ema200:
        return "PUT"
    return None

def format_signal_text(pair: str, direction: str, probability: int, snap: dict, atr_pct: float):
    entry = datetime.now() + timedelta(seconds=ENTRY_DELAY_SECONDS)
    exit_ = entry + timedelta(minutes=EXPIRATION_MINUTES)

    arrow = "📈" if direction == "CALL" else "📉"
    dir_word = "CALL" if direction == "CALL" else "PUT"

    return (
        f"📊 СИГНАЛ {pair}\n"
        f"{arrow} Направление: {dir_word}\n"
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

def is_owner(update: Update) -> bool:
    try:
        return update.effective_user and update.effective_user.id == OWNER_ID
    except Exception:
        return False


# ===================== ОБНОВЛЕНИЕ СТАТИСТИКИ =====================

async def apply_result(context: ContextTypes.DEFAULT_TYPE, result: str, trade_id: str, source: str = "AUTO"):
    """
    result: "WIN" | "LOSS"
    """
    global LOSS_STREAK, GLOBAL_PAUSE_UNTIL

    t = TRADES.get(trade_id)
    if not t or t.get("resolved"):
        return

    t["resolved"] = True
    t["result"] = result
    t["result_source"] = source

    if result == "WIN":
        STATS["win"] += 1
        DAY_STATS["win"] += 1
        LOSS_STREAK = 0
    else:
        STATS["loss"] += 1
        DAY_STATS["loss"] += 1
        LOSS_STREAK += 1
        if LOSS_STREAK == 1:
            GLOBAL_PAUSE_UNTIL = datetime.now() + timedelta(minutes=PAUSE_AFTER_1_LOSS_MIN)
        elif LOSS_STREAK == 2:
            GLOBAL_PAUSE_UNTIL = datetime.now() + timedelta(minutes=PAUSE_AFTER_2_LOSS_MIN)
        elif LOSS_STREAK >= 3:
            GLOBAL_PAUSE_UNTIL = datetime.now() + timedelta(minutes=PAUSE_AFTER_3_LOSS_MIN)

    total = STATS["win"] + STATS["loss"]
    winrate = calc_winrate(STATS["win"], STATS["loss"])

    pause_txt = ""
    if GLOBAL_PAUSE_UNTIL and datetime.now() < GLOBAL_PAUSE_UNTIL:
        pause_txt = f"\n⏸ Пауза до: {GLOBAL_PAUSE_UNTIL.strftime('%H:%M:%S')}"

    pair = t["pair"]
    direction = t["direction"]
    entry_price = t["entry_price"]
    exit_price = t.get("exit_price")

    emoji = "✅" if result == "WIN" else "❌"
    txt = (
        f"{emoji} РЕЗУЛЬТАТ {pair}\n"
        f"Направление: {direction}\n"
        f"Вход: {entry_price:.5f}\n"
        + (f"Выход: {exit_price:.5f}\n" if isinstance(exit_price, (int, float)) else "")
        + f"Итог: {result} ({source})\n\n"
        f"📊 Общая статистика: WIN {STATS['win']} / LOSS {STATS['loss']} (WR {winrate}%)\n"
        f"LOSS подряд: {LOSS_STREAK}{pause_txt}"
    )

    # Ответить в тред к сигналу, если есть message_id
    try:
        if t.get("message_id"):
            await context.bot.send_message(
                chat_id=CHANNEL_ID,
                text=txt,
                reply_to_message_id=t["message_id"]
            )
        else:
            await context.bot.send_message(chat_id=CHANNEL_ID, text=txt)
    except Exception as e:
        logger.warning("Failed to send result msg: %s", e)


# ===================== АВТООЦЕНКА (ПОСЛЕ ЭКСПИРАЦИИ) =====================

async def job_evaluate_trade(context: ContextTypes.DEFAULT_TYPE):
    job_data = context.job.data or {}
    trade_id = job_data.get("trade_id")
    if not trade_id:
        return

    t = TRADES.get(trade_id)
    if not t or t.get("resolved"):
        return

    pair = t["pair"]
    direction = t["direction"]
    entry_price = float(t["entry_price"])

    df, err = get_market_data(pair, interval="1min", outputsize=5)
    if err or df is None or len(df) < 2:
        logger.info("eval skip %s: %s", pair, err)
        return

    exit_price = float(df["close"].iloc[-1])
    t["exit_price"] = exit_price

    if direction == "CALL":
        result = "WIN" if exit_price > entry_price else "LOSS"
    else:
        result = "WIN" if exit_price < entry_price else "LOSS"

    await apply_result(context, result, trade_id, source="AUTO(TwelveData)")


# ===================== КНОПКИ WIN/LOSS (ручной override) =====================

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Только OWNER может фиксировать результаты, чтобы подписчики не ломали статистику
    if not query.from_user or query.from_user.id != OWNER_ID:
        try:
            await query.answer("Только владелец может нажимать эти кнопки.", show_alert=True)
        except Exception:
            pass
        return

    data = query.data or ""
    if "|" not in data:
        return

    action, trade_id = data.split("|", 1)
    if action not in ("WIN", "LOSS"):
        return

    # Ручной результат = приоритет
    await apply_result(context, action, trade_id, source="MANUAL")


# ===================== КОМАНДЫ (ТОЛЬКО OWNER) =====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return

    await update.message.reply_text(
        "🤖 Бот активен.\n"
        f"Автосигналы идут в канал {CHANNEL_NAME} (1 лучшая пара каждые 3 минуты при сильном сигнале).\n"
        f"После каждой сделки бот публикует WIN/LOSS автоматически (по TwelveData).\n"
        f"Ежедневный отчёт: {REPORT_HOUR:02d}:{REPORT_MINUTE:02d}\n\n"
        "Команды (только ты):\n"
        "/test — тест в канал\n"
        "/stats — статистика\n"
        "/report_now — отправить дневной отчёт сейчас\n"
        "/pulse_on — включить пульс\n"
        "/pulse_off — выключить пульс"
    )

async def test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    await update.message.reply_text("✅ Тест: отправляю сообщение в канал...")
    await context.bot.send_message(chat_id=CHANNEL_ID, text="✅ ТЕСТ: бот может писать в канал (OK)")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return

    winrate = calc_winrate(STATS["win"], STATS["loss"])
    pause_txt = ""
    if GLOBAL_PAUSE_UNTIL and datetime.now() < GLOBAL_PAUSE_UNTIL:
        pause_txt = f"\n⏸ Пауза до: {GLOBAL_PAUSE_UNTIL.strftime('%H:%M:%S')}"

    await update.message.reply_text(
        f"📊 Статистика (общая)\n"
        f"WIN: {STATS['win']}\n"
        f"LOSS: {STATS['loss']}\n"
        f"Winrate: {winrate}%\n"
        f"LOSS подряд: {LOSS_STREAK}{pause_txt}\n\n"
        f"📅 Сегодня\n"
        f"Сделок: {DAY_STATS['trades']}\n"
        f"WIN: {DAY_STATS['win']} / LOSS: {DAY_STATS['loss']}"
    )

async def report_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    await send_daily_report(context, reset_after=False)
    await update.message.reply_text("✅ Дневной отчёт отправлен в канал.")

async def pulse_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    global PULSE_ENABLED
    PULSE_ENABLED = True
    await update.message.reply_text("✅ Пульс включён (раз в 10 минут сообщение в канал)")

async def pulse_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    global PULSE_ENABLED
    PULSE_ENABLED = False
    await update.message.reply_text("✅ Пульс выключён")


# ===================== АВТООТЧЁТ =====================

async def send_daily_report(context: ContextTypes.DEFAULT_TYPE, reset_after: bool = True):
    win = DAY_STATS["win"]
    loss = DAY_STATS["loss"]
    trades = DAY_STATS["trades"]
    wr = calc_winrate(win, loss)

    best_pair = None
    best_prob = None
    if DAY_BEST:
        best_pair = max(DAY_BEST.items(), key=lambda x: x[1])[0]
        best_prob = DAY_BEST[best_pair]

    txt = (
        f"📊 {CHANNEL_NAME} | Дневной отчёт\n\n"
        f"📈 Сделок: {trades}\n"
        f"✅ WIN: {win}\n"
        f"❌ LOSS: {loss}\n"
        f"🎯 Winrate: {wr}%\n\n"
        + (f"🔥 Лучшая пара: {best_pair} (до {best_prob}%)\n" if best_pair else "🔥 Лучшая пара: —\n")
        + "\n🤖 Отчёт сформирован автоматически."
    )

    await context.bot.send_message(chat_id=CHANNEL_ID, text=txt)

    if reset_after:
        DAY_STATS["win"] = 0
        DAY_STATS["loss"] = 0
        DAY_STATS["trades"] = 0
        DAY_BEST.clear()


async def job_daily_report(context: ContextTypes.DEFAULT_TYPE):
    # Если выходные — отчёт тоже можно отправлять (по желанию). Оставляем включенным.
    await send_daily_report(context, reset_after=True)


# ===================== JOBS =====================

async def job_send_best_signal(context: ContextTypes.DEFAULT_TYPE):
    global GLOBAL_PAUSE_UNTIL, LAST_WEEKEND_NOTICE_DATE

    now = datetime.now()

    # Выходной режим
    if not is_market_open_now():
        today = now.strftime("%Y-%m-%d")
        if LAST_WEEKEND_NOTICE_DATE != today:
            LAST_WEEKEND_NOTICE_DATE = today
            await context.bot.send_message(
                chat_id=CHANNEL_ID,
                text=(
                    f"⏸ {CHANNEL_NAME} | Выходной режим\n\n"
                    "Рынок закрыт (Forex).\n"
                    "Анализ возобновится в понедельник."
                )
            )
        logger.info("Weekend mode: skip scanning")
        return

    # 4) глобальная пауза после лоссов
    if GLOBAL_PAUSE_UNTIL and now < GLOBAL_PAUSE_UNTIL:
        logger.info("GLOBAL PAUSE until %s", GLOBAL_PAUSE_UNTIL)
        return

    best = None  # (prob, pair, direction, snap, atr_pct)

    for pair in PAIRS:
        # 1) фильтр сессий
        if not in_session(pair):
            continue

        # анти-спам по паре
        last_time = LAST_SENT.get(pair)
        if last_time and (now - last_time).total_seconds() < COOLDOWN_MINUTES * 60:
            continue

        # 1m данные
        df1, err1 = get_market_data(pair, interval="1min", outputsize=300)
        if err1 or df1 is None or len(df1) < MIN_BARS_FOR_INDICATORS:
            logger.info("skip %s: %s", pair, err1)
            continue

        # 2) ATR фильтр + алерт "рынок ожил"
        atr_series = calculate_atr(df1, ATR_PERIOD)
        atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
        price = float(df1["close"].iloc[-1])
        atr_pct = (atr / price) * 100 if price else 0.0

        threshold = ATR_THRESHOLDS.get(pair, DEFAULT_ATR_PCT)
        was_low = LOWVOL_STATE.get(pair, False)

        if atr_pct < threshold:
            LOWVOL_STATE[pair] = True
            logger.info("low vol %s atr%%=%.3f (thr=%.3f)", pair, atr_pct, threshold)
            continue

        LOWVOL_STATE[pair] = False
        if was_low:
            await send_volatility_alert(context, pair, atr_pct, threshold)

        # 1m сигнал
        direction1, prob1, reason1, snap1 = build_signal_1m(df1)
        if direction1 is None:
            logger.info("no signal 1m %s: %s", pair, reason1)
            continue

        # 3) подтверждение 5m
        df5, err5 = get_market_data(pair, interval="5min", outputsize=300)
        if err5 or df5 is None or len(df5) < MIN_BARS_FOR_INDICATORS:
            logger.info("skip 5m %s: %s", pair, err5)
            continue

        direction5 = direction_confirm_5m(df5)
        if direction5 is None or direction5 != direction1:
            logger.info("mtf reject %s 1m=%s 5m=%s", pair, direction1, direction5)
            continue

        prob = min(90, prob1 + 3)  # бонус за MTF
        cand = (prob, pair, direction1, snap1, atr_pct)

        if best is None or cand[0] > best[0]:
            best = cand

    if best is None:
        logger.info("no strong signals this cycle")
        return

    prob, pair, direction, snap, atr_pct = best

    if prob < MIN_PROB_TO_SEND:
        logger.info("best prob=%s < %s, skip", prob, MIN_PROB_TO_SEND)
        return

    # Отправка сигнала
    text = format_signal_text(pair, direction, prob, snap, atr_pct)

    # trade_id
    trade_id = f"{pair}|{now.strftime('%Y%m%d%H%M%S')}"
    entry_time = now + timedelta(seconds=ENTRY_DELAY_SECONDS)
    exit_time = entry_time + timedelta(minutes=EXPIRATION_MINUTES)

    msg = await context.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        reply_markup=winloss_keyboard(trade_id)
    )

    # сохранить сделку
    TRADES[trade_id] = {
        "pair": pair,
        "direction": direction,
        "prob": prob,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": float(snap["price"]),
        "exit_price": None,
        "message_id": msg.message_id,
        "resolved": False,
    }

    # дневные метрики
    DAY_STATS["trades"] += 1
    DAY_BEST[pair] = max(DAY_BEST.get(pair, 0), prob)

    LAST_SENT[pair] = now
    logger.info("sent BEST: %s %s prob=%s", pair, direction, prob)

    # Запланировать авто-оценку WIN/LOSS
    delay = (exit_time - now).total_seconds() + EVAL_EXTRA_SECONDS
    if delay < 5:
        delay = 5

    context.job_queue.run_once(
        job_evaluate_trade,
        when=delay,
        data={"trade_id": trade_id},
        name=f"eval_{trade_id}"
    )


async def job_pulse(context: ContextTypes.DEFAULT_TYPE):
    if not PULSE_ENABLED:
        return

    if not is_market_open_now():
        await context.bot.send_message(chat_id=CHANNEL_ID, text=f"⏱ {CHANNEL_NAME}: выходные, рынок закрыт.")
        return

    await context.bot.send_message(chat_id=CHANNEL_ID, text=f"⏱ {CHANNEL_NAME}: бот жив, анализирую рынок...")


# ===================== MAIN =====================

def main():
    if not BOT_TOKEN or "PASTE_" in BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN не задан. Вставь токен в BOT_TOKEN или задай переменную окружения BOT_TOKEN.")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Команды (их увидишь только ты — остальные просто игнорируются)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("test", test))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("report_now", report_now))
    app.add_handler(CommandHandler("pulse_on", pulse_on))
    app.add_handler(CommandHandler("pulse_off", pulse_off))

    # Кнопки WIN/LOSS (нажимать может только OWNER)
    app.add_handler(CallbackQueryHandler(on_button))

    if app.job_queue is None:
        raise RuntimeError(
            "JobQueue не активен. Установи: python3 -m pip install 'python-telegram-bot[job-queue]'"
        )

    # Основной сканер сигналов
    app.job_queue.run_repeating(job_send_best_signal, interval=SIGNAL_INTERVAL_SECONDS, first=10)

    # Пульс
    app.job_queue.run_repeating(job_pulse, interval=600, first=60)

    # Ежедневный отчёт (локальная таймзона Mac)
    local_tz = datetime.now().astimezone().tzinfo
    report_time = time(hour=REPORT_HOUR, minute=REPORT_MINUTE, tzinfo=local_tz)
    app.job_queue.run_daily(job_daily_report, time=report_time, name="daily_report")

    print(f"🚀 {CHANNEL_NAME} BOT запущен: сигналы + авто WIN/LOSS + выходные + ежедневный отчёт {REPORT_HOUR:02d}:{REPORT_MINUTE:02d}")
    app.run_polling()


if __name__ == "__main__":
    main()
