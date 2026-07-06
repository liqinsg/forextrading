"""
Currency Strength Ranker — v2 (MA-based)
-----------------------------------------
Ranks 8 major currencies by relative strength using a Moving Average
comparison method rather than raw momentum.

For each pair on each timeframe:
  - Fetch N candles and compute a simple MA over the last MA_PERIOD closes
  - Score = (current_close - MA) / MA * 100  (% deviation from MA)
  - Attribute +score to base currency, -score to quote currency

This is more stable than raw momentum because:
  - MA smooths noise before comparison
  - A currency must be consistently above MA across MULTIPLE pairs to rank high
  - Short-term spikes don't flip rankings the way they do with raw % change

Timeframe weights: H8 > H4 >> H1 > M15
Short timeframes are included for early signal detection but carry low weight.
"""

import os
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

oanda_env    = os.getenv("OANDA_ENV", "practice")
oanda_client = API(access_token=os.getenv("OANDA_API_TOKEN"), environment=oanda_env)

# Full pair universe — covers all 8 currencies with enough cross-references
TRACKED_PAIRS = [
    # USD crosses
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY",
    # EUR crosses
    "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_CAD", "EUR_CHF",
    # GBP crosses
    "GBP_JPY", "GBP_AUD", "GBP_CAD",
    # Commodity crosses
    "AUD_JPY", "AUD_CAD", "AUD_CHF",
    "NZD_JPY", "CAD_JPY", "CHF_JPY",
]

CURRENCIES = ["USD", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "JPY"]

# Timeframe weights — heavier weight = stronger influence on final ranking.
# H8/H4 dominate so short-term noise doesn't flip rankings.
TIMEFRAMES = {
    "M15": 0,   # excluded — too noisy for swing trading
    "H1":  1,
    "H4":  3,
    "H8":  6,   # highest weight — best trend signal
}

MA_PERIOD    = 20   # MA period in candles per timeframe
CANDLE_COUNT = 30   # fetch slightly more than MA_PERIOD to ensure enough complete candles


def get_ma_score(instrument: str, granularity: str) -> float | None:
    """
    Returns % deviation of current close from its MA for this pair/timeframe.
    Positive = base currency is above MA (bullish for base, bearish for quote).
    Returns None on failure.
    """
    if TIMEFRAMES.get(granularity, 0) == 0:
        return None  # skip zero-weight timeframes entirely

    params = {"count": CANDLE_COUNT, "granularity": granularity}
    try:
        req = instruments.InstrumentsCandles(instrument=instrument, params=params)
        oanda_client.request(req)
        candles = [c for c in req.response.get("candles", []) if c["complete"]]
        if len(candles) < MA_PERIOD:
            return None

        closes = [float(c["mid"]["c"]) for c in candles]
        current = closes[-1]
        ma      = sum(closes[-MA_PERIOD:]) / MA_PERIOD

        # % deviation from MA — normalizes across pairs with very different price scales
        # e.g. USD/JPY at 162 vs EUR/USD at 1.13 — both expressed as % so comparable
        return (current - ma) / ma * 100

    except Exception as e:
        print(f"  [STRENGTH] {instrument} {granularity} failed: {e}")
        return None


def build_strength_matrix() -> dict[str, float]:
    """
    Computes weighted MA-based strength score for each of the 8 currencies.
    Higher score = price consistently above MA across more pairs = stronger.
    """
    scores        = {c: 0.0 for c in CURRENCIES}
    sample_counts = {c: 0   for c in CURRENCIES}

    for pair in TRACKED_PAIRS:
        base, quote = pair.split("_")
        if base not in CURRENCIES or quote not in CURRENCIES:
            continue

        for granularity, weight in TIMEFRAMES.items():
            if weight == 0:
                continue

            score = get_ma_score(pair, granularity)
            if score is None:
                continue

            # Positive MA score → base is strong, quote is weak
            scores[base]  += score * weight
            scores[quote] -= score * weight
            sample_counts[base]  += weight
            sample_counts[quote] += weight

    # Normalize by total weight so all currencies are on the same scale
    for c in CURRENCIES:
        if sample_counts[c] > 0:
            scores[c] /= sample_counts[c]

    return scores


def get_best_trade_candidate(scores: dict[str, float]) -> tuple[str, str, str] | tuple[None, None, None]:
    """
    Picks strongest vs weakest currency and finds the best tradeable pair.
    Returns (pair, action, rationale) or (None, None, None).
    """
    ranked   = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    strongest = ranked[0][0]
    weakest   = ranked[-1][0]

    score_gap = scores[strongest] - scores[weakest]

    # Three-tier classification based on score gap:
    #   COILING  (gap < 0.5)  — currencies bunched, no trend, wait for breakout
    #   MODERATE (gap 0.5-1.5) — some divergence, use regime-based routing
    #   STRONG   (gap > 1.5)  — clear winner/loser, enter immediately
    if score_gap < 0.5:
        print(f"  [STRENGTH] Score gap {score_gap:.3f} — COILING. "
              f"No trend signal. Route to S4 breakout-entry.")
        return None, "COILING", f"Score gap {score_gap:.3f} — currencies compressed, breakout pending"
    elif score_gap > 1.5:
        print(f"  [STRENGTH] Score gap {score_gap:.3f} — STRONG TREND. "
              f"Enter immediately without LLM gate.")
    else:
        print(f"  [STRENGTH] Score gap {score_gap:.3f} — MODERATE. "
              f"Routing to regime-based strategy.")

    # Try direct pair first
    candidate_a = f"{strongest}_{weakest}"
    candidate_b = f"{weakest}_{strongest}"

    if candidate_a in TRACKED_PAIRS:
        return (candidate_a, "BUY",
                f"{strongest} strongest ({scores[strongest]:+.4f}), "
                f"{weakest} weakest ({scores[weakest]:+.4f})")
    elif candidate_b in TRACKED_PAIRS:
        return (candidate_b, "SELL",
                f"{strongest} strongest ({scores[strongest]:+.4f}), "
                f"{weakest} weakest ({scores[weakest]:+.4f})")

    # Proxy: strongest vs next-weakest available pair
    for _, weak_currency in reversed(ranked):
        if weak_currency == strongest:
            continue
        ca = f"{strongest}_{weak_currency}"
        cb = f"{weak_currency}_{strongest}"
        if ca in TRACKED_PAIRS:
            return (ca, "BUY",
                    f"Proxy: {strongest} vs {weak_currency} "
                    f"(direct {weakest} pair unavailable)")
        elif cb in TRACKED_PAIRS:
            return (cb, "SELL",
                    f"Proxy: {strongest} vs {weak_currency} "
                    f"(direct {weakest} pair unavailable)")

    return None, None, None


# Score gap thresholds
SCORE_GAP_COILING = 0.5   # below this → coiling, wait for breakout
SCORE_GAP_STRONG  = 1.5   # above this → strong trend, enter immediately


def analyze_currency_strength() -> tuple[str, str, str, dict]:
    """
    Main entry point.
    Returns (pair_to_trade, action, rationale, scores_dict).

    action can be:
      "BUY" / "SELL"  — tradeable signal
      "COILING"       — currencies compressed, route to S4 stop orders
    Score gap > SCORE_GAP_STRONG signals a strong trend: enter without LLM gate.
    """
    print("[STRENGTH] Computing MA-based currency strength (H1×1, H4×3, H8×6)...")
    scores = build_strength_matrix()
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    gap    = ranked[0][1] - ranked[-1][1]

    if gap < SCORE_GAP_COILING:
        state = "COILING"
    elif gap > SCORE_GAP_STRONG:
        state = "STRONG"
    else:
        state = "MODERATE"

    print(f"\n  Currency Strength Ranking (gap={gap:.3f} → {state}):")
    for i, (currency, score) in enumerate(ranked, 1):
        bar_len = min(int(abs(score) * 20), 40)
        bar = "█" * bar_len
        direction = "▲" if score > 0 else "▼"
        print(f"  {i}. {currency}: {score:+.4f} {direction} {bar}")

    if state == "COILING":
        pair, _, _ = get_best_trade_candidate(scores)
        rationale  = (f"Score gap {gap:.3f} < {SCORE_GAP_COILING} — "
                      f"currencies compressed. Breakout pending. Route to S4 stops.")
        print(f"  → COILING: no directional trade. S4 stop orders recommended.")
        return pair, "COILING", rationale, scores

    pair, action, rationale = get_best_trade_candidate(scores)

    if pair:
        print(f"\n  Best candidate: {action} {pair}")
        if state == "STRONG":
            print(f"  → STRONG TREND (gap={gap:.3f} > {SCORE_GAP_STRONG}). Enter immediately.")
        print(f"  Rationale: {rationale}")
    else:
        print("\n  No tradeable pair found.")

    return pair, action, rationale, scores


def format_strength_for_llm(scores: dict, pair: str, action: str, rationale: str) -> str:
    """Formats strength matrix as a string for LLM context."""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    lines  = ["=== CURRENCY STRENGTH MATRIX (MA-based, H8×6 / H4×3 / H1×1) ==="]
    for i, (c, s) in enumerate(ranked, 1):
        lines.append(f"  #{i} {c}: {s:+.4f}")
    lines.append(f"\nStrength-based candidate: {action} {pair}")
    lines.append(f"Rationale: {rationale}")
    return "\n".join(lines)