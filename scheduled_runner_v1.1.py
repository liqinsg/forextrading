"""
Scheduled Runner — Simple JPY Trend Strategy
=============================================
Checks USD/JPY, EUR/JPY, GBP/JPY on each invocation.
Entry when H4 + H1 + M30 + M15 are ALL above MA5.
SL = today's low − 23 pips. TP = entry + 100 pips.

CRON MODE: this script runs ONE cycle and exits — no internal
scheduling loop. Interval control lives entirely in crontab, not here.
See the bottom of this file for the crontab line to add.

A file lock prevents two cycles overlapping if OANDA/network calls run
long and cron fires again before the previous cycle finished — without
it, overlapping runs could both see "no open position" and double-enter
the same pair.
"""
import fcntl
import os
import sys
from datetime import datetime

from config import CHECK_INTERVAL_MINUTES, RISK_LEVEL, RISK_PROFILE
from custom_strategy import analyze_custom_strategy, get_last_signal
from utils import execute_market_trade, get_open_position
from utils.schemas import TradeSignal
from retry import with_retry

LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scheduled_runner.lock")


def run_cycle():
    profile = RISK_PROFILE[RISK_LEVEL]
    print(f"\n[{datetime.now().isoformat()}] === JPY Trend Scan | Risk: {RISK_LEVEL} ===")
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
        try:
            existing = get_open_position(pair)
        except Exception as e:
            print(f"  [NETWORK ERROR] OANDA connection dropped: {e}. Will retry next cron tick.")
            return  # exit cleanly; cron fires again on its own schedule

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


def main():
    print("=== JPY Trend Runner (cron mode) ===")
    print("  Pairs    : USD/JPY, EUR/JPY, GBP/JPY")
    print("  Signal   : H4+H1+M30+M15 all above MA5")
    print("  SL       : Today's low − 23 pips")
    print("  TP       : Entry + 100 pips")
    print(f"  Risk     : Level {RISK_LEVEL} ({RISK_PROFILE[RISK_LEVEL]['units']:,} units)")
    print(f"  Interval : every {CHECK_INTERVAL_MINUTES} min — set in crontab, not this script")

    # Non-blocking exclusive lock: if a previous cycle is still running
    # (stuck on a slow OANDA call, retries, etc.), skip this tick rather
    # than starting a second overlapping cycle.
    lock_file = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        print(f"[{datetime.now().isoformat()}] Previous cycle still running (lock held). Skipping this tick.")
        sys.exit(0)

    try:
        run_cycle()
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


if __name__ == "__main__":
    main()

# ==========================================
# CRONTAB — add via `crontab -e`
# ==========================================
# Run every CHECK_INTERVAL_MINUTES (15 shown below — must match config.py's
# CHECK_INTERVAL_MINUTES manually; the two are no longer auto-synced since
# cron now owns the schedule instead of this script's old while-loop).
#
# */15 * * * * cd /home/qili/projects/gemini_api && /home/qili/miniconda3/envs/ai-sprint/bin/python3 scheduled_runner.py >> logs/scheduled_runner.log 2>&1
#
# Notes:
#   - Use the ai-sprint env's full python3 path directly (as above) rather
#     than `conda activate ai-sprint && python3 ...` — cron runs a
#     non-login, non-interactive shell that doesn't source conda's shell
#     hooks, so `conda activate` silently fails or picks the wrong env.
#   - Adjust the path if you're back on ansible81 (nie) vs. this machine
#     (qili) — the two hosts have different home directories.
#   - `logs/` must already exist in the project directory (it does).