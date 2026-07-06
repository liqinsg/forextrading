import importlib
from config import OANDA_ACCOUNT_ID
from main import oanda_client


def get_support_resistance(
    instrument: str,
    granularity: str = "H1",
    count: int = 100,
    window: int = 3,
):
    """
    Returns nearest support and resistance.

    Example:
        {
            "support": 215.116,
            "resistance": 216.076
        }
    """

    instruments_module = importlib.import_module(
        "oandapyV20.endpoints.instruments"
    )

    params = {
        "count": count,
        "granularity": granularity
    }

    req = instruments_module.InstrumentsCandles(
        instrument=instrument,
        params=params
    )

    oanda_client.request(req)

    candles = [
        c for c in req.response["candles"]
        if c["complete"]
    ]

    lows = [float(c["mid"]["l"]) for c in candles]
    highs = [float(c["mid"]["h"]) for c in candles]
    closes = [float(c["mid"]["c"]) for c in candles]

    current_price = closes[-1]

    supports = []
    resistances = []

    for i in range(window, len(candles) - window):

        low = lows[i]
        high = highs[i]

        if low == min(lows[i-window:i+window+1]):
            supports.append(low)

        if high == max(highs[i-window:i+window+1]):
            resistances.append(high)

    support = max(
        [s for s in supports if s <= current_price],
        default=min(lows)
    )

    resistance = min(
        [r for r in resistances if r >= current_price],
        default=max(highs)
    )

    return {
        "support": support,
        "resistance": resistance,
        "current_price": current_price
    }