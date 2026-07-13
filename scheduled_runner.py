"""
Scheduled Runner — Simple JPY Trend Strategy
=============================================
Checks USD/JPY, EUR/JPY, GBP/JPY every 15 minutes.
Entry when H4 + H1 + M30 + M15 are ALL above MA5.
SL = today's low − 23 pips. TP = entry + 100 pips.
No AI/LLM calls. Pure rules.
"""
import time
import schedule
from datetime import datetime

from config import (
    CHECK_INTERVAL_MINUTES, RISK_LEVEL, RISK_PROFILE
)
from custom_strategy import analyze_custom_strategy, get_last_signal
from utils import execute_market_trade, get_open_position  # , run_trading_cycle, get_candles, get_latest_price
from utils.schemas import TradeSignal
from retry import with_retry


def run_cycle():
    profile = RISK_PROFILE[RISK_LEVEL]
    print(
        f"\n[{datetime.now().isoformat()}] === JPY Trend Scan | Risk: {RISK_LEVEL} ===")

    try:
        # 1. Scan all three JPY pairs for MA5 alignment
        with_retry(analyze_custom_strategy, max_attempts=3, delay=5, label="strategy_scan")

        signal_data = get_last_signal()

        if signal_data is None:
            print("[CYCLE] No MA5 alignment across any JPY pair. HOLD.")
            return

        pair = signal_data["pair"]
        action = signal_data["action"]

        # 2. Guard: skip if already in this pair
        # Inside scheduled_runner.py -> run_cycle()
        try:
            existing = get_open_position(pair)
        except Exception as e:
            print(
                f"  [NETWORK ERROR] OANDA connection dropped: {e}. Retrying next cycle...")
            return  # Safely abort this 15-min cycle without crashing the daemon script
        if existing:
            lu = float(existing.get("long", {}).get("units", 0))
            su = float(existing.get("short", {}).get("units", 0))
            if lu != 0 or su != 0:
                print(f"[CYCLE] Already holding {pair}. Skipping new entry.")
                return

        # 3. Build TradeSignal and execute
        signal = TradeSignal(
            pair_to_trade=pair,
            action=action,
            confidence_score=0.85,   # rule-based, high confidence
            stop_loss=signal_data["stop_loss"],
            take_profit=signal_data["take_profit"],
            reasoning=signal_data["reasoning"]
        )

        print(f"\n  SIGNAL    : {action} {pair}")
        print(f"  ENTRY     : {signal_data['entry']}")
        print(f"  SL        : {signal.stop_loss}")
        print(f"  TP        : {signal.take_profit}")
        print(f"  REASONING : {signal.reasoning}")
        print("\n[CYCLE] Executing order...")

        execute_market_trade(signal, units_override=profile["units"])

    except Exception as e:
        import traceback
        print(f"[CYCLE ERROR] {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("=== JPY Trend Runner ===")
    print("  Pairs    : USD/JPY, EUR/JPY, GBP/JPY")
    print("  Signal   : H4+H1+M30+M15 all above MA5")
    print("  SL       : Today's low − 23 pips")
    print("  TP       : Entry + 100 pips")
    print(
        f"  Risk     : Level {RISK_LEVEL} ({RISK_PROFILE[RISK_LEVEL]['units']:,} units)")
    print(f"  Interval : every {CHECK_INTERVAL_MINUTES} min")
    print("  Press Ctrl+C to stop\n")

    run_cycle()
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(run_cycle)
    while True:
        schedule.run_pending()
        time.sleep(5)
