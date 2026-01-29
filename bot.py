async def job_expiry_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data or {}
    signal_id = data.get("signal_id")
    if not signal_id:
        return

    trade = OPEN_TRADES.get(signal_id)
    if not trade:
        return

    symbol = trade["symbol"]
    entry_price = float(trade["entry_price"])
    direction = trade["direction"]

    # если уже был посчитан результат — не дублируем
    if signal_id in TRADE_RESULTS:
        return

    try:
        exit_price = td_quote_price(symbol)
    except Exception:
        await post_to_channel(
            context,
            f"⏱ Экспирация прошла по *{symbol}*.\n"
            f"⚠️ Не смог проверить цену (лимит/ошибка API).\n"
            f"Отметь вручную кнопкой *WIN/LOSS* под сигналом.\n"
            f"🆔 id: `{signal_id}`"
        )
        return

    # движение
    if exit_price > entry_price:
        move = "⬆️ ВВЕРХ"
    elif exit_price < entry_price:
        move = "⬇️ ВНИЗ"
    else:
        move = "➡️ ФЛЭТ"

    # авто итог
    result = "flat"
    auto_text = "➡️ ФЛЭТ"
    if move != "➡️ ФЛЭТ":
        if direction == "CALL" and exit_price > entry_price:
            result = "win"
            auto_text = "✅ WIN (авто, по API)"
        elif direction == "PUT" and exit_price < entry_price:
            result = "win"
            auto_text = "✅ WIN (авто, по API)"
        else:
            result = "loss"
            auto_text = "❌ LOSS (авто, по API)"

    # запис
