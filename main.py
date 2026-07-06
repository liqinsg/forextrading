from config import OANDA_ENV, OANDA_API_TOKEN
from oandapyV20 import API
from google import genai
from google.genai import types
from custom_strategy import analyze_custom_strategy
from models_ensemble import get_gemini_decision, get_qwen_decision, get_deepseek_decision, use_qwen, use_deepseek
from config import OANDA_ACCOUNT_ID, OANDA_ENV, OANDA_API_TOKEN
import importlib
from breaking_entry import detect_protracted_range, place_breakout_stop_orders

gemini_client = genai.Client()

try:
    orders = importlib.import_module("oandapyV20.endpoints.orders")
except ImportError:
    class _OrdersFallback:
        class OrderCreate:
            def __init__(self, *args, **kwargs):
                raise ImportError("oandapyV20 is not installed")
    orders = _OrdersFallback()

# Single shared OANDA client — loaded from shared config

oanda_client = API(access_token=OANDA_API_TOKEN, environment=OANDA_ENV)


def format_price_for_instrument(price, instrument: str) -> str:
    """Round prices to the precision required by OANDA for the given instrument."""
    try:
        numeric_price = float(price)
    except (TypeError, ValueError):
        return str(price)

    if instrument.endswith("_JPY"):
        return f"{numeric_price:.3f}"
    return f"{numeric_price:.5f}"


def get_open_position(instrument: str):
    positions_module = importlib.import_module(
        "oandapyV20.endpoints.positions"
    )
    account_id = OANDA_ACCOUNT_ID
    request = positions_module.OpenPositions(accountID=account_id)
    oanda_client.request(request)

    for pos in request.response.get("positions", []):
        if pos.get("instrument") == instrument:
            return pos

    return None


def attach_sl_tp_to_open_trade(signal, instrument: str | None = None) -> bool:
    """Attach or replace SL/TP on the current open trade for the given instrument."""
    instrument = instrument or signal.pair_to_trade
    position = get_open_position(instrument)
    if not position:
        print(
            f"[EXECUTION] No open position found for {instrument}; cannot attach SL/TP.")
        return False

    trade_ids = []
    for side in (position.get("long", {}), position.get("short", {})):
        trade_ids.extend(side.get("tradeIDs", []))

    if not trade_ids:
        print(
            f"[EXECUTION] Open position exists for {instrument} but has no trade IDs.")
        return False

    trade_id = trade_ids[0]
    account_id = OANDA_ACCOUNT_ID
    trades_module = importlib.import_module("oandapyV20.endpoints.trades")

    payload = {
        "stopLoss": {
            "price": format_price_for_instrument(signal.stop_loss, instrument),
            "timeInForce": "GTC",
            "triggerMode": "TOP_OF_BOOK"
        },
        "takeProfit": {
            "price": format_price_for_instrument(signal.take_profit, instrument),
            "timeInForce": "GTC"
        }
    }

    try:
        account_id = OANDA_ACCOUNT_ID
        request = trades_module.TradeCRCDO(
            accountID=account_id, tradeID=trade_id, data=payload)
        oanda_client.request(request)
        print(
            f"[EXECUTION] SL/TP attached to trade {trade_id} for {instrument}.")
        verify_sl_tp_on_trade(trade_id, instrument)
        return True
    except Exception as e:
        print(
            f"[EXECUTION ERROR] Could not attach SL/TP to trade {trade_id}: {e}")
        return False


def verify_sl_tp_on_trade(trade_id: str, instrument: str) -> None:
    """Fetch the opened trade details and explicitly confirm attached SL/TP orders."""
    account_id = OANDA_ACCOUNT_ID
    try:
        trades_module = importlib.import_module("oandapyV20.endpoints.trades")
        req = trades_module.TradeDetails(
            accountID=account_id, tradeID=trade_id)
        oanda_client.request(req)
        trade = req.response.get("trade", {})

        sl_order = trade.get("stopLossOrder", {})
        tp_order = trade.get("takeProfitOrder", {})
        if sl_order or tp_order:
            print(f"[EXECUTION VERIFY] Trade {trade_id} brackets confirmed:")
            if sl_order:
                print(
                    f"  SL order ID {sl_order.get('id')} @ {sl_order.get('price')}")
            if tp_order:
                print(
                    f"  TP order ID {tp_order.get('id')} @ {tp_order.get('price')}")
        else:
            print(
                f"[EXECUTION VERIFY] WARNING — Trade {trade_id} has no SL/TP. Check HUB manually.")

    except Exception as e:
        print(
            f"[EXECUTION VERIFY ERROR] Could not verify trade {trade_id}: {e}")


def get_recent_range(instrument: str, granularity: str = "H1", lookback: int = 20) -> tuple[float, float, float] | None:
    """Returns (top, bottom, current_close) for the recent range on the given instrument."""
    instruments_module = importlib.import_module(
        "oandapyV20.endpoints.instruments")
    params = {"count": lookback + 1, "granularity": granularity}
    try:
        request = instruments_module.InstrumentsCandles(
            instrument=instrument, params=params)
        oanda_client.request(request)
        candles = [c for c in request.response.get(
            "candles", []) if c["complete"]]
        if len(candles) < lookback:
            return None

        highs = [float(c["mid"]["h"]) for c in candles[:-1]]
        lows = [float(c["mid"]["l"]) for c in candles[:-1]]
        current_close = float(candles[-1]["mid"]["c"])
        return max(highs), min(lows), current_close
    except Exception as e:
        print(f"[RANGE] Could not fetch range for {instrument}: {e}")
        return None


def place_range_limit_orders(instrument: str, units: int, lookback: int = 20):
    """Places both buy limit and sell limit orders at the recent range bottom and top."""
    account_id = OANDA_ACCOUNT_ID
    order_ids = []
    range_data = get_recent_range(instrument, lookback=lookback)
    if not range_data:
        print(
            f"[RANGE] Unable to compute recent range for {instrument}. No limit orders placed.")
        return order_ids

    top_price, bottom_price, current_price = range_data
    top_price_str = format_price_for_instrument(top_price, instrument)
    bottom_price_str = format_price_for_instrument(bottom_price, instrument)

    for price, direction in [(bottom_price_str, "BUY"), (top_price_str, "SELL")]:
        order_units = str(units if direction == "BUY" else -units)
        payload = {
            "order": {
                "instrument": instrument,
                "units": order_units,
                "price": price,
                "type": "LIMIT",
                "timeInForce": "GTC",
                "positionFill": "DEFAULT",
                "clientExtensions": {
                    "comment": "Range-entry test order",
                    "tag": "range-limit-test"
                }
            }
        }

        try:
            request = orders.OrderCreate(accountID=account_id, data=payload)
            oanda_client.request(request)
            order_txn = request.response.get("orderCreateTransaction", {})
            order_id = order_txn.get("id") or request.response.get(
                "relatedTransactionIDs", [None])[0]
            print(
                f"[RANGE] Placed {direction} LIMIT for {instrument} at {price} (order id {order_id})")
            order_ids.append(order_id)
        except Exception as e:
            print(
                f"[RANGE] Failed to place {direction} LIMIT for {instrument} at {price}: {e}")

    return order_ids


def place_bottom_range_buy_order(instrument: str, units: int, lookback: int = 20):
    """Places a single buy limit order at the recent range bottom."""
    account_id = OANDA_ACCOUNT_ID
    range_data = get_recent_range(instrument, lookback=lookback)
    if not range_data:
        print(
            f"[RANGE] Unable to compute recent range for {instrument}. No bottom buy order placed.")
        return []

    _, bottom_price, current_price = range_data
    price = format_price_for_instrument(bottom_price, instrument)
    payload = {
        "order": {
            "instrument": instrument,
            "units": str(units),
            "price": price,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "positionFill": "DEFAULT",
            "clientExtensions": {
                "comment": "Range-bottom buy test order",
                "tag": "range-bottom-buy-test"
            }
        }
    }

    try:
        request = orders.OrderCreate(accountID=account_id, data=payload)
        oanda_client.request(request)
        order_txn = request.response.get("orderCreateTransaction", {})
        order_id = order_txn.get("id") or request.response.get(
            "relatedTransactionIDs", [None])[0]
        print(
            f"[RANGE] Placed BUY LIMIT for {instrument} at {price} (order id {order_id})")
        return [order_id]
    except Exception as e:
        print(
            f"[RANGE] Failed to place BUY LIMIT for {instrument} at {price}: {e}")
        return []


def execute_market_trade(signal, units_override=None):
    """
    Translates an AI TradeSignal into a market order with SL/TP brackets.
    units_override: used by the risk-level scheduler to scale position size.
    """
    if signal is None or signal.action == "HOLD":
        print("[EXECUTION] No actionable signal. No orders placed.")
        return

    # Guard: don't stack on top of an existing open position
    existing = get_open_position(signal.pair_to_trade)
    if existing:
        long_units = float(existing.get("long", {}).get("units", 0))
        short_units = float(existing.get("short", {}).get("units", 0))
        if long_units != 0 or short_units != 0:
            print(f"[EXECUTION] Skipping — already have an open position on "
                  f"{signal.pair_to_trade} (long={long_units}, short={short_units}).")
            return

    # Fetch live price and validate SL/TP are on the correct sides before submitting.
    # This catches bad LLM outputs before OANDA rejects them with a cryptic error.
    try:
        pricing_module = importlib.import_module(
            "oandapyV20.endpoints.pricing")
        account_id_for_price = OANDA_ACCOUNT_ID
        price_req = pricing_module.PricingInfo(
            accountID=account_id_for_price,
            params={"instruments": signal.pair_to_trade}
        )
        oanda_client.request(price_req)
        price_data = price_req.response["prices"][0]
        current_ask = float(price_data["asks"][0]["price"])
        current_bid = float(price_data["bids"][0]["price"])
        print(
            f"[PRICE CHECK] {signal.pair_to_trade} — bid: {current_bid} | ask: {current_ask}")

        if signal.action == "BUY":
            if signal.stop_loss >= current_ask:
                print(
                    f"[PRICE CHECK] REJECTED — BUY stop_loss ({signal.stop_loss}) >= ask ({current_ask}). Would fill at a loss.")
                return
            if signal.take_profit <= current_ask:
                print(
                    f"[PRICE CHECK] REJECTED — BUY take_profit ({signal.take_profit}) <= ask ({current_ask}). Would fill at a loss.")
                return
        elif signal.action == "SELL":
            if signal.stop_loss <= current_bid:
                print(
                    f"[PRICE CHECK] REJECTED — SELL stop_loss ({signal.stop_loss}) <= bid ({current_bid}). Would fill at a loss.")
                return
            if signal.take_profit >= current_bid:
                print(
                    f"[PRICE CHECK] REJECTED — SELL take_profit ({signal.take_profit}) >= bid ({current_bid}). Would fill at a loss.")
                return
        print(f"[PRICE CHECK] SL/TP bracket validated OK.")

    except Exception as e:
        print(
            f"[PRICE CHECK] Could not fetch live price ({e}). Proceeding with caution.")

    account_id = OANDA_ACCOUNT_ID
    base_units = units_override if units_override is not None else 10000
    units = base_units if signal.action == "BUY" else -base_units

    order_payload = {
        "order": {
            "units": str(units),
            "instrument": signal.pair_to_trade,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": format_price_for_instrument(signal.stop_loss, signal.pair_to_trade)},
            "takeProfitOnFill": {"price": format_price_for_instrument(signal.take_profit, signal.pair_to_trade)},
            "clientExtensions": {
                # OANDA hard limit is 128 chars
                "comment": signal.reasoning[:128],
                "tag": "ai-ensemble-strategy"
            }
        }
    }

    try:
        request = orders.OrderCreate(accountID=account_id, data=order_payload)
        oanda_client.request(request)
        response = request.response

        fill_txn = response.get("orderFillTransaction")
        cancel_txn = response.get("orderCancelTransaction")

        if fill_txn:
            print("\n================ ORDER FILLED ================")
            print(f"  Pair      : {signal.pair_to_trade}")
            print(f"  Direction : {signal.action}")
            print(f"  Units     : {units:,}")
            print(f"  Fill Price: {fill_txn.get('price')}")
            print(f"  Txn ID    : {fill_txn.get('id')}")
            print(f"  SL        : {signal.stop_loss}")
            print(f"  TP        : {signal.take_profit}")
            print("==============================================")
            attach_sl_tp_to_open_trade(signal)
        elif cancel_txn:
            print(f"\n[ORDER CANCELLED] Reason: {cancel_txn.get('reason')}")
        else:
            print("\n[WARNING] No fill or cancel in response — check HUB manually.")

    except Exception as e:
        print(f"[EXECUTION ERROR] Order routing failed: {e}")


def get_latest_news_sentiment() -> str:
    """
    Fetches LIVE macro FX sentiment via Gemini grounded on real-time Google Search.
    Replaces the old hardcoded static string.
    """
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="""Search for today's latest FX and macro news. Summarize in 4-6 sentences
            covering: (1) Bank of Japan policy stance and any JPY intervention signals,
            (2) USD strength/weakness drivers, (3) any G10 central bank surprises or major
            risk events this week. Be factual and concise — this is context for an algorithmic
            FX trading system.""",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
            )
        )
        sentiment = response.text.strip()
        print(f"[SENTIMENT] Live news fetched ({len(sentiment)} chars)")
        return sentiment
    except Exception as e:
        print(f"[SENTIMENT] Live fetch failed ({e}), using fallback.")
        return (
            "No live sentiment available this cycle. "
            "Apply extra caution and prefer HOLD if technical signals are ambiguous."
        )


def get_ensemble_consensus(prompt: str):
    """
    Queries all configured models. Returns a signal only if every responding
    model agrees on both pair and direction. Returns (signal_or_None, all_signals).
    """
    signals = []

    try:
        signals.append(("gemini", get_gemini_decision(prompt)))
    except Exception as e:
        print(f"[ENSEMBLE] Gemini call failed: {e}")

    if use_qwen:
        try:
            signals.append(("qwen", get_qwen_decision(prompt)))
        except Exception as e:
            print(f"[ENSEMBLE] Qwen call failed: {e}")

    if use_deepseek:
        try:
            signals.append(("deepseek", get_deepseek_decision(prompt)))
        except Exception as e:
            print(f"[ENSEMBLE] DeepSeek call failed: {e}")

    if not signals:
        print("[ENSEMBLE] No model returned a usable signal.")
        return None, signals

    first_pair = signals[0][1].pair_to_trade
    first_action = signals[0][1].action
    agreed = all(
        s.pair_to_trade == first_pair and s.action == first_action
        for _, s in signals
    )

    if not agreed:
        print("[ENSEMBLE] Models disagree:")
        for name, s in signals:
            print(
                f"  {name}: {s.action} {s.pair_to_trade} (confidence {s.confidence_score:.0%})")
        return None, signals

    return signals[0][1], signals


def run_trading_cycle():
    """Manual single-cycle run (used when calling main.py directly)."""
    print("\n=============================================")
    print("[SYSTEM] Executing Multi-Pair Matrix Strategy...")
    print("=============================================")

    signal = None
    try:
        macro_matrix = analyze_custom_strategy()
        current_sentiment = get_latest_news_sentiment()

        prompt = f"""
        You are an elite quantitative currency portfolio coordinator.
        Evaluate the live market data and sentiment below.

        {macro_matrix}

        --- LIVE MACROECONOMIC SENTIMENT ---
        {current_sentiment}
        """

        print("[3/3] Parsing through ensemble analysis...")
        signal, all_signals = get_ensemble_consensus(prompt)

        if signal is None:
            print("[SYSTEM] No consensus reached. No action this cycle.")
            return

        print("\n================ DECISION METRICS ================")
        print(f"  ASSET     : {signal.pair_to_trade}")
        print(f"  DIRECTION : {signal.action}")
        print(f"  CONFIDENCE: {signal.confidence_score * 100:.1f}%")
        print(f"  TP        : {signal.take_profit}")
        print(f"  SL        : {signal.stop_loss}")
        print(f"  REASONING : {signal.reasoning}")
        print("==================================================")

    except Exception as e:
        print(f"[ERROR] Strategy execution failed: {e}")
        return

    execute_market_trade(signal)


if __name__ == "__main__":
    run_trading_cycle()
