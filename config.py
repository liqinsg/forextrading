"""
Central configuration — edit this file to control all strategy behaviour.
Do not hardcode these values elsewhere in the codebase.
"""
import os
from dotenv import load_dotenv
load_dotenv()
# ==========================================
# OANDA connection
# ==========================================
OANDA_ENV = os.getenv("OANDA_ENV", "practice")
OANDA_API_TOKEN = os.getenv("OANDA_API_TOKEN", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")

# ==========================================
# Scheduler
# ==========================================
CHECK_INTERVAL_MINUTES = 15   # 15 min matches the fastest signal timeframe (M15)
REQUIRE_ALIGNED = 4
MIN_VALID_PAIRS_TO_TRADE = 2   # must have ≥2 valid JPY crosses to trade the top one
# ==========================================
# ACTIVE STRATEGY SETTINGS
# Simple MA5 multi-timeframe trend strategy on JPY pairs.
# ==========================================

# Pairs to actually trade (direct orders placed here)
# TRADE_PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "USD_CHF", "EUR_CHF", "GBP_CHF", "AUD_CHF"]
# TRADE_PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
TRADE_PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY", "CAD_JPY"]
# Timeframes that must ALL be above MA5 for an entry signal
SIGNAL_TIMEFRAMES = ["H4", "H1", "M30", "M15"]

# TP / SL in pips (JPY pairs: 1 pip = 0.01)
TP_PIPS = 100   # fixed take profit: entry + 100 pips
SL_BUFFER_PIPS = 20    # pips below today's daily low
SPREAD_PIPS = 3     # conservative spread buffer added to SL

# ==========================================
# RISK LEVEL  (1 = safest, 10 = most aggressive)
# Controls position size per trade.
# ==========================================
RISK_LEVEL = 10

RISK_PROFILE = {
    1: {"units": 1000, "min_confidence": 0.90},
    2: {"units": 2000, "min_confidence": 0.85},
    3: {"units": 3000, "min_confidence": 0.80},
    4: {"units": 4000, "min_confidence": 0.75},
    5: {"units": 5000, "min_confidence": 0.70},
    6: {"units": 6000, "min_confidence": 0.65},
    7: {"units": 7000, "min_confidence": 0.60},
    8: {"units": 8000, "min_confidence": 0.55},
    9: {"units": 9000, "min_confidence": 0.50},
    10: {"units": 10000, "min_confidence": 0.40},
}

# ==========================================
# AI / GEMINI  (disabled for simple strategy)
# Set True to re-enable LLM validation when needed.
# Gemini model used for grounded news lookups, and a fallback tried once if
# the primary hits a quota/429 error. Verify both are still valid/available
# on your API key/tier periodically -- model names and free-tier quotas on
# Gemini's side change independently of this codebase.
# ==========================================
USE_GEMINI_AI = False
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_NEWS_MODEL = "gemini-3.5-flash"
GEMINI_NEWS_FALLBACK_MODEL = "gemini-flash-lite-latest"

# ==========================================
# ADVANCED STRATEGY SETTINGS
# Used by the full scheduled_runner.py (not the simple runner).
# Safe to ignore when running scheduled_runner_simple.py.
# ==========================================

# Meta selector
META_MODE = "MANUAL"
MANUAL_STRATEGY = 1     # 1=TREND_COMBINED, 2=RANGE_REVERSION,
# 3=BREAKOUT_CONFIRM, 4=TREND_PULLBACK
STRATEGY_PULLBACK = 4

# ADX regime thresholds
ADX_TREND_THRESHOLD = 25
ADX_BREAKOUT_THRESHOLD = 20

# ATR-based SL/TP (used by full runner, not simple runner)
ATR_GRANULARITY = "D"
ATR_CANDLE_COUNT = 20
ATR_MULTIPLIER_SL = 0.75
ATR_MULTIPLIER_TP = 2.0

# Range strategy
RANGE_LOOKBACK = 20
RANGE_TP_RATIO = 0.6
RANGE_SL_RATIO = 0.25

# Breaking / coiling entry
BREAKOUT_DURATION_HOURS = 4
BREAKOUT_WIDTH_PCT = 2.0

# Test mode — forces a specific pair regardless of signal
FORCE_TEST_PAIR = False
TEST_PAIR = None    # e.g. "USD_JPY"

# Order expiry for pending limit/stop orders
EXPIRE_AFTER = 1440       # minutes (1 day)

# ==========================================
# JPY TREND STRATEGY (custom_strategy.py)
# --------------------------------------------------------------------------
# Everything below was previously hardcoded directly inside
# custom_strategy.py and did NOT read from this file -- even the several
# names that already existed here (MIN_QUALIFYING_PAIRS, STRENGTH_*,
# CURRENCIES, ENABLE_EMA_TREND, ENABLE_NEWS_FILTER just below) had a second,
# separate hardcoded copy inside custom_strategy.py that actually got used.
# Editing those values here previously had NO EFFECT on the live strategy.
# custom_strategy.py now imports everything in this section directly, so
# this file is the single source of truth again, as the module docstring
# above always intended.
# ==========================================

# Minimum number of candidate JPY-cross pairs that must independently pass
# every filter (proportional strength cutoff, MA5/EMA alignment, news,
# structural S/R, min R:R) in a single cycle before ANY trade is taken.
# ------------------------------------------------------------------------
MIN_QUALIFYING_PAIRS = 2

STRENGTH_CANDLE_COUNT = 10  # superseded by STRENGTH_FAST_LOOKBACK/STRENGTH_SLOW_LOOKBACK
                            # below -- no longer read by custom_strategy.py;
                            # kept only in case something else still imports it.
STRENGTH_TIMEFRAMES = {
    "H1": 1,
    "H4": 3,
    "H8": 6,
}
STRENGTH_PAIRS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD",
    "USD_CAD", "USD_JPY",
    "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_CAD",
    "GBP_JPY", "GBP_AUD", "GBP_CAD",
    "AUD_JPY", "AUD_CAD", 
    "NZD_JPY", "CAD_JPY"
]

CURRENCIES = ["USD", "EUR", "GBP", "AUD", "NZD", "CAD", "JPY"]

# --- Blended fast/slow strength momentum -----------------------------------
# STRENGTH_TIMEFRAMES (H1/H4/H8, weighted 1/3/6) are all LONGER than the
# M15/M30/H1/H4 entry alignment check -- momentum over STRENGTH_CANDLE_COUNT
# bars of H8 covers several days, so a fresh price reversal shows up in the
# MA5/EMA entry check long before it shows up in strength. That mismatch is
# what produces cases like "currency strength says BUY, all 4 entry
# timeframes say SELL" for extended periods (observed on GBP_JPY/EUR_JPY,
# 2026-07-10 scan). Blending a fast lookback with the existing slow one
# makes strength react on a similar timescale to the entry check, without
# discarding longer-term context entirely.
STRENGTH_FAST_LOOKBACK = 5     # bars -- reacts quickly to fresh reversals
STRENGTH_SLOW_LOOKBACK = 20    # bars -- retains prior longer-term context
STRENGTH_FAST_WEIGHT = 0.7
STRENGTH_SLOW_WEIGHT = 0.3

# Acceleration = fast_momentum - slow_momentum. Positive acceleration means
# a currency is strengthening faster than its longer-term trend suggests
# (or a bearish trend is decelerating); negative means the opposite. Added
# on top of the blended score to flag turning points earlier. Defaults to
# False to keep behavior simpler/more tested; flip on to compare.
ENABLE_STRENGTH_ACCELERATION = False
STRENGTH_ACCELERATION_WEIGHT = 0.5

# ATR period used only for ENABLE_ATR_NORMALIZED_STRENGTH's normalization
# of the blended momentum figure below (separate from JPY_ATR_PERIOD /
# DOMINANCE_ATR_PERIOD, which serve different, unrelated calculations).
STRENGTH_ATR_PERIOD = 14

# --- Strategy refinement switches -----------------------------------------
# Each defaults to False so the strategy behaves exactly as already tested.
# Flip individually to compare old vs. new behavior; safe to combine.

# False = original close>MA5 alignment check (fast, sensitive to small retraces).
# True  = EMA10>EMA20 trend alignment per timeframe instead (smoother, less
# prone to churn on a 20-pip retrace inside a larger trend).
ENABLE_EMA_TREND = False

# False = raw % price momentum for currency strength (current behavior).
# True  = momentum normalized by ATR, so a 0.5% move in a low-volatility pair
# isn't weighted the same as a 0.5% move in a high-volatility one.
# NOTE: this changes the *scale* of strength scores -- MIN_MARKET_STRENGTH
# and the dynamic 40% cutoff were tuned for the raw % scale. Re-tune those
# after switching this on; don't assume the same thresholds still make sense.
ENABLE_ATR_NORMALIZED_STRENGTH = False

# False = "price traded beyond daily resistance/support" triggers a weekly
# target (current behavior) -- can fire on a single intraday wick.
# True  = requires BREAKOUT_CONFIRMATION_CLOSES consecutive completed daily
# closes beyond the level before treating it as a genuine breakout.
ENABLE_BREAKOUT_CONFIRMATION = False
BREAKOUT_CONFIRMATION_CLOSES = 2

# False = SL/TP anchored to nearest daily/weekly structural support/resistance
# (current default). True = SL/TP distance instead scales with the pair's own
# recent ATR(JPY_ATR_PERIOD) -- wider automatically in high-volatility
# conditions, tighter in calm ones -- independent of where the nearest S/R
# line happens to sit.
ENABLE_ATR_SLTP = False

# --- News filter (Gemini + Google Search grounding) ------------------------
# Master on/off switch for the economic-news guard. Flip to False to disable
# news checking entirely and run the strategy exactly as before -- useful
# for backtests, or if Gemini is ever unavailable/quota-exhausted.
ENABLE_NEWS_FILTER = False

# Every high-impact event Gemini returns for a traded pair gets appended
# here, whether or not it actually triggers a skip -- a running reference
# log of what news context was around at each scan cycle.
NEWS_LOG_PATH = "news_events.log"

# Currencies checked for high-impact events (covers all configured JPY
# crosses' base currencies plus JPY itself).
NEWS_CURRENCIES = ["USD", "JPY", "EUR", "GBP"]

# --- Signal corroboration (directional consensus across JPY crosses) ------
# How much more combined |strength_score| weight the dominant direction
# (BUY vs SELL) must have over the opposing direction to be treated as a
# real consensus rather than a narrow majority. A bare headcount majority
# (e.g. 2 BUY pairs at +0.3 each vs 1 SELL pair at -1.8) is NOT enough on
# its own -- the opposing side may carry more actual conviction even with
# fewer pairs. 1.5 means the dominant side's combined weight must be at
# least 50% larger than the opposing side's combined weight.
MIN_DOMINANCE_RATIO = 1.5

# When True, each pair's strength_score is divided by that pair's own
# ATR(DOMINANCE_ATR_PERIOD) (daily candles) before being summed into the
# BUY/SELL dominance weights -- so a raw +2.4 gap on a naturally choppy
# pair (e.g. GBP_JPY) doesn't automatically outweigh a raw +1.4 gap on a
# typically calmer pair (e.g. AUD_JPY) just because of a scale mismatch in
# "how big a move is normal" for each instrument. Does NOT change which
# specific pair is finally selected within the winning direction -- only
# which direction (or whether any direction) is allowed to trade at all.
ENABLE_VOLATILITY_NORMALIZED_DOMINANCE = False
DOMINANCE_ATR_PERIOD = 14

# --- JPYTrendStrategy trade parameters --------------------------------------
JPY_PIP = 0.01
MIN_MARKET_STRENGTH = 0.05
FRONT_RUN_PIPS = 15
MACRO_PROTECTION_PIPS = 20
MIN_RR = 1.2   # minimum reward:risk ratio required to keep a signal

# --- JPYTrendStrategy ATR-based SL/TP (only used when ENABLE_ATR_SLTP=True) -
# NOTE: deliberately named JPY_ATR_* rather than reusing ATR_MULTIPLIER_SL/
# ATR_MULTIPLIER_TP above -- those are marked as belonging to a different
# ("full") runner, and this repo doesn't show what else might read them, so
# they're kept separate here rather than silently repurposed.
JPY_ATR_PERIOD = 14
JPY_ATR_HISTORY_LOOKBACK = 50          # prior ATR readings defining "normal" volatility
JPY_ATR_SL_MULTIPLIER_NORMAL = 2.2
JPY_ATR_SL_MULTIPLIER_HIGH_VOL = 2.8   # used when current ATR's z-score > 1 (unusually volatile)
JPY_ATR_SL_MULTIPLIER_LOW_VOL = 1.8    # used when current ATR's z-score < -1 (unusually calm)
JPY_ATR_RR_MULTIPLE = 2.0              # TP distance = SL distance * this (bakes in R:R directly)

OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")

# ==========================================
# DATA PROVIDER SETTINGS
# Control where price/candle data comes from
# ==========================================
# DATA_SOURCE = "OANDA_WITH_YAHOO_FALLBACK"   # Options:
#                                             # "OANDA_ONLY"
#                                             # "YAHOO_ONLY"
#                                             # "OANDA_WITH_YAHOO_FALLBACK"
# ==========================================
# DATA PROVIDER SETTINGS
# Control where price/candle data comes from
# ==========================================
DATA_SOURCE = "OANDA_WITH_YAHOO_FALLBACK"
# --- ML Confirmation Layer (custom_strategy.py / ml_confirmation.py) ---
ENABLE_ML_CONFIRMATION = False        # gate signals out on low ML confidence
ENABLE_ML_WEIGHTED_DOMINANCE = False  # scale dominance weight by ML confidence
ML_MIN_CONFIDENCE = 0.55
ML_RETRAIN_HOURS = 24
ML_TRAIN_GRANULARITY = "H1"
ML_TRAIN_CANDLE_COUNT = 3000
ML_HOLDOUT_FRACTION = 0.2
ML_LABEL_HORIZON = 3
ML_MIN_HOLDOUT_F1 = 0.0

# --- SIDEWAYS / RANGE DETECTION (TEST MODE) ---
ENABLE_RANGE_DETECTOR = True       # Set to True to enable & compare
RANGE_DETECT_LOOKBACK_DAYS = 5
RANGE_DETECT_MAX_RANGE_PCT = 1.2
RANGE_DETECT_MIN_VOL_RATIO = 0.8
SKIP_SIDEWAYS_PAIRS = True  # False = ignore sideways check, trade all pairs regardless of range

TRADE_TOP_PAIRS = 1  # Always trade only single strongest/weakest pair per cycle