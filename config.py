"""
Central configuration — edit this file to control all strategy behaviour.
Do not hardcode these values elsewhere in the codebase.
"""
import os

from joblib import expires_after

# ==========================================
# OANDA connection
# ==========================================
OANDA_ENV        = os.getenv("OANDA_ENV", "practice")
OANDA_API_TOKEN  = os.getenv("OANDA_API_TOKEN", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")

# ==========================================
# Scheduler
# ==========================================
CHECK_INTERVAL_MINUTES = 1

# ==========================================
# META SELECTOR
# Controls who decides which strategy to use each cycle.
#   "RULES_SELECTS" — ADX threshold routes automatically (recommended)
#   "AI_SELECTS"    — Gemini reviews regime data and picks the strategy
#   "MANUAL"        — always use MANUAL_STRATEGY below, no regime detection
# ==========================================
# META_MODE       = "RULES_SELECTS"
META_MODE = "RULES_SELECTS"
STRATEGY_PULLBACK = 4
MANUAL_STRATEGY = 4   # only used when META_MODE = "MANUAL"
                      # 1=TREND_COMBINED, 2=RANGE_REVERSION, 3=BREAKOUT_CONFIRM

# ==========================================
# REGIME THRESHOLDS (ADX-based)
# ==========================================
ADX_TREND_THRESHOLD    = 25   # ADX above this → TRENDING  → use S1 Trend
ADX_BREAKOUT_THRESHOLD = 20   # ADX between this and trend  → BREAKOUT watch → use S3
                               # ADX below breakout          → RANGING  → use S2 Range

# ==========================================
# RISK LEVEL  (1 = safest, 10 = most aggressive)
# Controls position size and minimum confidence threshold.
# ==========================================
RISK_LEVEL = 10

RISK_PROFILE = {
    1:  {"units": 1000,  "min_confidence": 0.90},
    2:  {"units": 2000,  "min_confidence": 0.85},
    3:  {"units": 3000,  "min_confidence": 0.80},
    4:  {"units": 4000,  "min_confidence": 0.75},
    5:  {"units": 5000,  "min_confidence": 0.70},
    6:  {"units": 6000,  "min_confidence": 0.65},
    7:  {"units": 7000,  "min_confidence": 0.60},
    8:  {"units": 8000,  "min_confidence": 0.55},
    9:  {"units": 9000,  "min_confidence": 0.50},
    10: {"units": 10000, "min_confidence": 0.40},
}

# ==========================================
# ATR SETTINGS
# Use H4 candles for swing-trade-appropriate stop distances.
# SL multiplier 1.5× ATR, TP multiplier 2.5× ATR → R:R 1:1.67
# ==========================================
# ATR source: Daily candles give ~150-200 pip range on GBP/JPY —
# much more appropriate than H4 (~43 pip) which is noise territory.
ATR_GRANULARITY   = "D"    # Daily ATR for swing-appropriate stops
ATR_CANDLE_COUNT  = 20     # 20 daily candles = 4 weeks of data

# SL: 0.75× daily ATR ≈ 120-150 pips on GBP/JPY — survives intraday noise
# TP: 2.0× daily ATR ≈ 320-400 pips on GBP/JPY — targets the real move
# R:R = 1:2.67 — significantly better than the previous 1:1.67
ATR_MULTIPLIER_SL = 0.75
ATR_MULTIPLIER_TP = 2.0

# ==========================================
# RANGE STRATEGY SETTINGS
# ==========================================
RANGE_LOOKBACK   = 20      # number of H1 candles to define the range box
RANGE_TP_RATIO   = 0.6     # take profit at 60% of range width (mean reversion)
RANGE_SL_RATIO   = 0.25    # stop loss at 25% beyond range boundary

USE_GEMINI_AI = False  # if True, Gemini AI will review the regime and confirm the trade

BREAKOUT_DURATION_HOURS = 4
BREAKOUT_WIDTH_PCT = 2.0

FORCE_TEST_PAIR = False
TEST_PAIR = None  # e.g. "GBP_JPY" — if set, only this pair will be considered for trading

EXPIRE_AFTER = 1440  # one day minutes