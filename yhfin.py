import argparse
import json

from indicator_provider import summarize_indicator_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch symbol data and compute local indicators.")
    parser.add_argument("symbol", nargs="?", default="SPY", help="Ticker symbol to fetch (default SPY)")
    parser.add_argument("--period", default="60d", help="History lookback period for yfinance")
    parser.add_argument("--interval", default="1d", help="Data interval for yfinance")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = summarize_indicator_data(args.symbol, period=args.period, interval=args.interval)
    print(json.dumps(result))
