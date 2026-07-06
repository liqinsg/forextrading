import os
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

from indicator_provider import fetch_price_data, safe_column, translate_pair_to_yahoo, oanda_granularity_to_yf_interval

# Dynamically fetch env; defaults to 'practice' for maximum safety
oanda_env = os.getenv("OANDA_ENV", "practice")
oanda_token = os.getenv("OANDA_API_TOKEN")

# Initialize client cleanly using the dynamic target
oanda_client = API(access_token=oanda_token, environment=oanda_env)

# oanda_client = API(access_token=os.getenv("OANDA_API_TOKEN"), environment="live")


def get_ma_trend(instrument: str, granularity: str, period: int = 20) -> str:
    """Calculates whether current price is above or below a moving average for a timeframe."""
    yahoo_symbol = translate_pair_to_yahoo(instrument)
    if yahoo_symbol:
        interval = oanda_granularity_to_yf_interval(granularity)
        try:
            df = fetch_price_data(instrument, period="7d", interval=interval)
            closes = safe_column(df, "Close").dropna()
            if len(closes) >= period + 1:
                current_close = closes.iloc[-1]
                ma = closes.tail(period).mean()
                return "UPWARDS" if current_close > ma else "DOWNWARDS"
        except Exception as e:
            print(
                f"[STRATEGY] yfinance trend fetch failed for {instrument} {granularity}: {e}")

    params = {"count": period + 5, "granularity": granularity}
    try:
        request = instruments.InstrumentsCandles(
            instrument=instrument, params=params)
        oanda_client.request(request)
        candles = request.response.get('candles', [])
        if len(candles) >= period:
            closes = [float(c['mid']['c']) for c in candles if c['complete']]
            current_close = closes[-1]
            ma = sum(closes[-period:]) / period
            return "UPWARDS" if current_close > ma else "DOWNWARDS"
    except Exception:
        return "UNKNOWN"
    return "UNKNOWN"


def analyze_custom_strategy() -> str:
    print("[STRATEGY] Scanning expanded matrix and multi-timeframe trends...")

    # Track the expanded basket
    pairs = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY"]
    matrix_report = "=== EXPANDED CROSS-CURRENCY ANALYSIS MATRIX ===\n"

    for pair in pairs:
        # Fetch multi-timeframe alignment
        m15_trend = get_ma_trend(pair, "M15")
        h1_trend = get_ma_trend(pair, "H1")
        h4_trend = get_ma_trend(pair, "H4")
        h8_trend = get_ma_trend(pair, "H8")

        matrix_report += f"""
        {pair} Alignment Checklist:
        - 15-Min Trend: {m15_trend}
        - 1-Hour Trend: {h1_trend}
        - 4-Hour Trend: {h4_trend}
        - 8-Hour Trend: {h8_trend}
        """

    matrix_report += """
    STRICT TREND ALIGNMENT RULES:
    1. DO NOT issue a SHORT signal on any pair if the 15M, 1H, or 8H timeframes are still pointing UPWARDS. A 4H dip alone is a trap.
    2. Review all timeframes across the asset list to isolate the single weakest currency cross (e.g., if AUD/JPY shows a downward shift across multiple timeframes compared to USD/JPY, target AUD/JPY instead).
    3. If timeframes conflict significantly, default the action to 'HOLD'.
    """
    return matrix_report
