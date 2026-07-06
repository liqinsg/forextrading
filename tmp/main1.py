import os
from oandapyV20 import API
from custom_strategy import analyze_custom_strategy
from models_ensemble import get_gemini_decision, get_qwen_decision, get_deepseek_decision, use_qwen, use_deepseek
import importlib
from dotenv import load_dotenv
load_dotenv()

try:
    orders = importlib.import_module("oandapyV20.endpoints.orders")
except ImportError:
    class _OrdersFallback:
        class OrderCreate:
            def __init__(self, *args, **kwargs):
                raise ImportError("oandapyV20 is not installed")

    orders = _OrdersFallback()

# Single, shared OANDA client. Respects OANDA_ENV (defaults to "practice").
oanda_env = os.getenv("OANDA_ENV", "practice")
oanda_client = API(access_token=os.getenv("OANDA_API_TOKEN"), environment=oanda_env)


def format_price_for_instrument(price, instrument: str) -> str:
    """Round prices to the precision required by OANDA for the given instrument."""
    try:
        numeric_price = float(price)
    except (TypeError, ValueError):
        return str(price)

    if instrument.endswith("_JPY"):
        return f"{numeric_price:.3f}"
    return f"{numeric_price:.5f}"


def execute_market_trade(signal):
    """
    Translates an AI TradeSignal payload into a structural market order
    complete with linked Take Profit and Stop Loss protections.
    """
    if signal is None or signal.action == "HOLD":
        print("[EXECUTION] No actionable signal. No orders placed.")
        return

    account_id = os.getenv("OANDA_ACCOUNT_ID")

    # Establish trade position unit sizing (Positive numbers = BUY, Negative = SELL)
    units = 10000 if signal.action == "BUY" else -10000

    order_payload = {
        "order": {
            "units": str(units),
            "instrument": signal.pair_to_trade,
            "timeInForce": "FOK",  # Fill Or Kill
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": format_price_for_instrument(signal.stop_loss, signal.pair_to_trade)},
            "takeProfitOnFill": {"price": format_price_for_instrument(signal.take_profit, signal.pair_to_trade)}
        }
    }

    try:
        request = orders.OrderCreate(accountID=account_id, data=order_payload)
        oanda_client.request(request)
        print("\n================ ORDER FILLED ================")
        print(f"Successfully entered {signal.action} position on {signal.pair_to_trade}!")
        print(f"Order Details Transaction ID: {request.response.get('orderFillTransaction', {}).get('id')}")
        print("==============================================")
    except Exception as e:
        print(f"[EXECUTION ERROR] Order routing failed: {e}")


def get_latest_news_sentiment() -> str:
    return (
        "Markets are anticipating safe-haven inflows into the Japanese Yen following global trade discussions. "
        "Eurozone indicators hint at stagnating growth figures, while the British Pound remains resilient due to hawkish rate talk."
    )


def get_ensemble_consensus(prompt: str):
    """
    Queries all available models and only returns a signal if every
    *configured* provider agrees on both pair and action. Providers without
    an API key are skipped rather than treated as a disagreement.

    Returns (signal_or_None, list_of_individual_signals).
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

    # Require unanimous agreement on pair + action across everything that responded.
    first_pair = signals[0][1].pair_to_trade
    first_action = signals[0][1].action
    agreed = all(
        s.pair_to_trade == first_pair and s.action == first_action
        for _, s in signals
    )

    if not agreed:
        print("[ENSEMBLE] Models disagree:")
        for name, s in signals:
            print(f"  - {name}: {s.action} {s.pair_to_trade}")
        return None, signals

    return signals[0][1], signals


def run_trading_cycle():
    print("\n=============================================")
    print("[SYSTEM] Executing Multi-Pair Matrix Strategy...")
    print("=============================================")

    signal = None

    try:
        # 1. Gather Strategy Matrix Inputs
        macro_matrix = analyze_custom_strategy()
        current_sentiment = get_latest_news_sentiment()

        # 2. Construct the Matrix Prompt
        prompt = f"""
        You are an elite quantitative currency portfolio coordinator operating under 'custom_strategy'.
        Evaluate the live market data matrix and fundamental sentiment details provided below.

        {macro_matrix}

        --- LIVE MACROECONOMIC SENTIMENT ---
        {current_sentiment}
        """

        # 3. Query consensus across configured models
        print("[3/3] Parsing matrix through ensemble analysis...")
        signal, all_signals = get_ensemble_consensus(prompt)

        if signal is None:
            print("[SYSTEM] No consensus reached. Defaulting to no action this cycle.")
            return

        # 4. Display the decision metrics
        print("\n================ DECISION METRICS ================")
        print(f"ASSET TARGET     : {signal.pair_to_trade}")
        print(f"RECOMMENDED DIR  : {signal.action}")
        print(f"CONFIDENCE METRIC: {signal.confidence_score * 100:.1f}%")
        print(f"TARGET TAKE PROFIT: {signal.take_profit}")
        print(f"PROTECTIVE STOP  : {signal.stop_loss}")
        print(f"SYSTEM LOGIC     : {signal.reasoning}")
        print("==================================================")

    except Exception as e:
        print(f"[ERROR] Strategy execution failed: {e}")
        return

    # 5. Execute the order on the linked OANDA account, only if we got a valid signal
    execute_market_trade(signal)


if __name__ == "__main__":
    run_trading_cycle()