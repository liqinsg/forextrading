"""
Scheduled Trading Runner — v2
==============================
Each cycle:
  1. Currency strength matrix finds strongest vs weakest pair
  2. Regime detector classifies market (TRENDING / RANGING / BREAKOUT_WATCH)
  3. Meta selector routes to the right strategy for that regime
  4. Chosen strategy generates a TradeSignal (direction + ATR-based SL/TP)
  5. Risk gate checks confidence, position guard, price validity
  6. OANDA execution

META_MODE in config.py controls step 3:
  RULES_SELECTS — ADX threshold decides strategy automatically
  AI_SELECTS    — Gemini reviews regime data and picks the strategy
  MANUAL        — always use MANUAL_STRATEGY from config.py

Edit config.py to change behaviour. Don't hardcode values here.
"""

from stop_orders import place_stop_entry_order
import os
import time
# from turtle import st
import schedule
from datetime import datetime, timezone, timedelta
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments_ep
from google import genai
from google.genai import types
from main import orders
from find_support_resistence import get_support_resistance
# from strategy_pullback import run_trend_pullback
import oandapyV20.endpoints.orders as orders_ep

from config import (
    OANDA_API_TOKEN, OANDA_ENV, OANDA_ACCOUNT_ID,
    CHECK_INTERVAL_MINUTES, RISK_LEVEL, RISK_PROFILE,
    META_MODE, MANUAL_STRATEGY,
    ADX_TREND_THRESHOLD, ADX_BREAKOUT_THRESHOLD,
    ATR_GRANULARITY, ATR_CANDLE_COUNT,
    ATR_MULTIPLIER_SL, ATR_MULTIPLIER_TP,
    RANGE_LOOKBACK, RANGE_TP_RATIO, RANGE_SL_RATIO,
    BREAKOUT_DURATION_HOURS, BREAKOUT_WIDTH_PCT,
    STRATEGY_PULLBACK, USE_GEMINI_AI, EXPIRE_AFTER,
    FORCE_TEST_PAIR, TEST_PAIR
)

from main import (
    execute_market_trade, get_latest_news_sentiment,
    get_ensemble_consensus, get_open_position,
    place_bottom_range_buy_order, get_recent_range,
    format_price_for_instrument
)
from retry import with_retry
from quota_guard import quota_guard
from custom_strategy import analyze_custom_strategy
from currency_strength import analyze_currency_strength, format_strength_for_llm
from regime_detector import (
    detect_regime, get_regime_context_for_llm,
    REGIME_TRENDING, REGIME_RANGING, REGIME_BREAKOUT
)
from utils.schemas import TradeSignal

STRATEGY_TREND = 1
STRATEGY_RANGE = 2
STRATEGY_BREAKOUT = 3

STRATEGY_LABELS = {
    1: "S1 Trend combined",
    2: "S2 Range reversion",
    3: "S3 Breakout confirm",
    4: "S4 Trend pullback",
}

oanda_client = API(access_token=OANDA_API_TOKEN, environment=OANDA_ENV)
gemini_client = genai.Client()


def has_pending_order(pair: str, tag: str | None = None) -> bool:
    req = orders_ep.OrderList(accountID=OANDA_ACCOUNT_ID)
    oanda_client.request(req)

    for order in req.response.get("orders", []):
        if order.get("instrument") != pair:
            continue

        if tag is None:
            return True

        client_ext = order.get("clientExtensions", {})
        if client_ext.get("tag") == tag:
            return True

    return False


def place_range_entry_order(
    pair: str,
    action: str,
    units: int,
    support: float,
    resistance: float,
    sl: float,
    tp: float,
):
    account_id = OANDA_ACCOUNT_ID
    # dp = 3 if pair.endswith("_JPY") else 5
    # breakout_buffer = 0.05 if pair.endswith("_JPY") else 0.0005

    if action == "BUY":
        orders_to_place = [
            ("LIMIT", support),
        ]
    else:
        orders_to_place = [
            ("LIMIT", resistance),
        ]

    failed = False
    for order_type, entry_price in orders_to_place:

        order_units = units if action == "BUY" else -units

        payload = {
            "order": {
                "instrument": pair,
                "units": str(order_units),
                "price": str(entry_price),
                "type": order_type,
                "timeInForce": "GTC",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": format_price_for_instrument(sl, pair)
                },
                "takeProfitOnFill": {
                    "price": format_price_for_instrument(tp, pair)
                },
                "clientExtensions": {
                    "comment": f"{action} {order_type}",
                    "tag": f"{action.lower()}-{order_type.lower()}"
                }
            }
        }

        try:
            req = orders.OrderCreate(
                accountID=account_id,
                data=payload
            )
            oanda_client.request(req)

            order_id = (
                req.response
                .get("orderCreateTransaction", {})
                .get("id")
            )

            print(
                f"[S2] {action} {order_type} "
                f"{pair} @ {entry_price} "
                f"(id={order_id})"
            )

        except Exception as e:
            failed = True
            print(
                f"[S2 ERROR] "
                f"{action} {order_type} "
                f"{pair}: {e}"
            )

    if failed:
        raise RuntimeError(f"One or more entry orders failed for {pair}")
# ==========================================
# ATR-BASED SL/TP  (H4 candles, configurable)
# ==================================


def estimate_sr_sl_tp(pair: str, action: str) -> tuple[float, float, float]:
    levels = get_support_resistance(pair)

    support = levels["support"]
    resistance = levels["resistance"]
    current_price = levels["current_price"]

    dp = 3 if pair.endswith("_JPY") else 5
    pip = 0.01 if pair.endswith("_JPY") else 0.0001
    buffer = pip * 10  # 2 pip buffer outside S/R

    if action == "BUY":
        sl = round(support - buffer, dp)
        tp = round(resistance - buffer, dp)

        if not sl < current_price < tp:
            raise ValueError(
                f"Invalid BUY S/R levels for {pair}: "
                f"SL={sl}, price={current_price}, TP={tp}, "
                f"support={support}, resistance={resistance}"
            )

    else:
        sl = round(resistance + buffer, dp)
        tp = round(support + buffer, dp)

        if not tp < current_price < sl:
            raise ValueError(
                f"Invalid SELL S/R levels for {pair}: "
                f"TP={tp}, price={current_price}, SL={sl}, "
                f"support={support}, resistance={resistance}"
            )

    print(
        f"  [S/R] {pair} | price: {current_price} | "
        f"support: {support} | resistance: {resistance} | "
        f"SL: {sl} | TP: {tp}"
    )

    return sl, tp, current_price


def estimate_sl_tp(pair: str, action: str) -> tuple[float, float, float]:
    """
    Returns (stop_loss, take_profit, current_price).
    Uses H4 candles for swing-appropriate stop distances (~40-60 pips on JPY pairs).
    Rounds to correct OANDA precision: 3dp for JPY pairs, 5dp for others.
    """
    params = {"count": ATR_CANDLE_COUNT + 5, "granularity": ATR_GRANULARITY}
    req = instruments_ep.InstrumentsCandles(instrument=pair, params=params)
    oanda_client.request(req)
    candles = [c for c in req.response.get("candles", []) if c["complete"]]

    if len(candles) < 5:
        raise ValueError(
            f"Not enough {ATR_GRANULARITY} candles for ATR on {pair}")

    trs = []
    for i in range(1, len(candles)):
        h, l, pc = float(candles[i]["mid"]["h"]), float(
            candles[i]["mid"]["l"]), float(candles[i-1]["mid"]["c"])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))

    atr = sum(trs[-14:]) / min(14, len(trs))
    current_price = float(candles[-1]["mid"]["c"])
    dp = 3 if pair.endswith("_JPY") else 5

    if action == "BUY":
        sl = round(current_price - atr * ATR_MULTIPLIER_SL, dp)
        tp = round(current_price + atr * ATR_MULTIPLIER_TP, dp)
    else:
        sl = round(current_price + atr * ATR_MULTIPLIER_SL, dp)
        tp = round(current_price - atr * ATR_MULTIPLIER_TP, dp)

    # atr_pips = atr * 100 if not pair.endswith("_JPY") else atr * 100
    print(f"  [ATR/{ATR_GRANULARITY}] {pair} | price: {current_price} | "
          f"ATR: {atr:.5f} | SL: {sl} | TP: {tp} | R:R 1:{ATR_MULTIPLIER_TP/ATR_MULTIPLIER_SL:.2f}")

    return sl, tp, current_price


# ==========================================
# META SELECTOR
# ==========================================
def select_strategy_by_rules(regime: str) -> int:
    """Deterministic routing based on ADX regime."""
    if regime == REGIME_TRENDING:
        return STRATEGY_TREND
    elif regime == REGIME_BREAKOUT:
        return STRATEGY_BREAKOUT
    else:
        return STRATEGY_RANGE


def select_strategy_by_ai(pair: str, regime: str, metrics: dict,
                          strength_context: str, sentiment: str) -> int:
    """Ask Gemini to pick the best strategy given full regime context."""
    regime_context = get_regime_context_for_llm(pair, regime, metrics)
    prompt = f"""
    You are a quantitative FX trading system coordinator.
    
    Based on the market regime analysis and macro sentiment below, select the
    most appropriate trading strategy for this cycle. Respond with ONLY a JSON
    object with a single field "strategy" containing an integer: 1, 2, or 3.
    
    Strategy options:
      1 = TREND_COMBINED   — suitable when ADX > 25, clear directional momentum
      2 = RANGE_REVERSION  — suitable when ADX < 20, price oscillating in a range
      3 = BREAKOUT_CONFIRM — suitable when ADX 20-25, price approaching range boundary
    
    {regime_context}
    
    {strength_context}
    
    --- LIVE SENTIMENT ---
    {sentiment}
    
    Return only: {{"strategy": <1, 2, or 3>}}
    """
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )
        import json
        result = json.loads(response.text)
        chosen = int(result.get("strategy", STRATEGY_TREND))
        if chosen not in STRATEGY_LABELS:
            chosen = STRATEGY_TREND
        print(
            f"  [AI META] Gemini selected strategy {chosen}: {STRATEGY_LABELS[chosen]}")
        return chosen
    except Exception as e:
        print(
            f"  [AI META] Strategy selection failed ({e}). Defaulting to rules-based.")
        return select_strategy_by_rules(regime)


# ==========================================
# STRATEGY 1: TREND COMBINED
# ==========================================
def run_trend_combined(pair: str, action: str, rationale: str,
                       scores: dict, profile: dict, sentiment: str,
                       skip_llm: bool = False) -> TradeSignal | None:
    """Currency strength candidate → LLM validates → ATR SL/TP."""
    try:
        # sl, tp, current_price = estimate_sl_tp(pair, action)
        sl, tp, current_price = estimate_sr_sl_tp(pair, action)
    except Exception as e:
        # print(f"  [S1] ATR failed: {e}")
        print(f"  [S1] S/R SL/TP failed: {e}")
        return None

    strength_context = format_strength_for_llm(scores, pair, action, rationale)
    macro_matrix = analyze_custom_strategy()

    prompt = f"""
    You are an elite quantitative FX portfolio coordinator.

    Currency strength matrix candidate: {action} {pair}
    Current live price: {current_price}
    (ATR-based SL/TP applied automatically — set stop_loss=0 take_profit=0 in response.)

    Validate this trade. Confirm (BUY/SELL) if justified, or respond HOLD if not.

    {strength_context}

    --- TECHNICAL MATRIX ---
    {macro_matrix}

    --- LIVE SENTIMENT ---
    {sentiment}
    """

    if skip_llm or not USE_GEMINI_AI or not quota_guard.is_available():
        print("  [S1] Skipping LLM validation, using rules signal.")
        return TradeSignal(
            pair_to_trade=pair,
            action=action,
            confidence_score=0.70,
            stop_loss=sl,
            take_profit=tp,
            reasoning=f"[QUOTA FALLBACK] Rules-only: {rationale}"
        )

    try:
        signal, _ = get_ensemble_consensus(prompt)
        quota_guard.record_call()
    except Exception as e:
        quota_guard.handle_error(e)
        print("  [S1] Gemini ensemble failed — falling back to rules signal.")
        return TradeSignal(
            pair_to_trade=pair,
            action=action,
            confidence_score=0.70,
            stop_loss=sl,
            take_profit=tp,
            reasoning=f"[QUOTA FALLBACK] Rules-only after error: {rationale}"
        )

    if signal is None or signal.action == "HOLD":
        return signal

    return TradeSignal(
        pair_to_trade=signal.pair_to_trade,
        action=signal.action,
        confidence_score=signal.confidence_score,
        stop_loss=sl,
        take_profit=tp,
        reasoning=signal.reasoning
    )


# ==========================================
# STRATEGY 2: RANGE REVERSION
# ==========================================
def run_range_reversion(pair: str, action: str, profile: dict) -> TradeSignal | None:
    """
    Places a limit order at the range bottom (BUY) or top (SELL).
    TP is set at RANGE_TP_RATIO of range width toward the midpoint.
    SL is set RANGE_SL_RATIO beyond the range boundary.
    """

    levels = get_support_resistance(pair)

    support = levels["support"]
    resistance = levels["resistance"]
    # current_price = levels["current_price"]

    top = resistance
    bottom = support

    range_width = top - bottom
    dp = 3 if pair.endswith("_JPY") else 5

    if action == "BUY":
        entry = bottom
        tp = round(bottom + range_width * RANGE_TP_RATIO, dp)
        sl = round(bottom - range_width * RANGE_SL_RATIO, dp)
    else:
        entry = top
        tp = round(top - range_width * RANGE_TP_RATIO, dp)
        sl = round(top + range_width * RANGE_SL_RATIO, dp)

    entry_str = format_price_for_instrument(entry, pair)
    print(
        f"  [S2] Range {bottom:.5f}–{top:.5f} | Entry: {entry_str} | SL: {sl} | TP: {tp}")

    # Place a limit order (not market) — wait for price to reach range boundary
    # account_id = OANDA_ACCOUNT_ID
    # order_units = profile["units"] if action == "BUY" else -profile["units"]

    try:
        place_range_entry_order(
            pair=pair,
            action=action,
            units=profile["units"],
            support=support,
            resistance=resistance,
            sl=sl,
            tp=tp,
        )
        return None
    except Exception as e:
        print(f"  [S2] Limit order failed: {e}")
        return None

# ==========================================
# STRATEGY 3: BREAKOUT CONFIRM
# ==========================================


def run_breakout_confirm(pair: str, action: str, metrics: dict,
                         profile: dict, sentiment: str) -> TradeSignal | None:
    """
    Confirms a breakout: price must have closed above/below range with
    momentum (ADX rising, +DI/-DI aligned). Enters market order on retest.
    """
    if not metrics:
        print("  [S3] No regime metrics available.")
        return None

    current_price = metrics.get("current_price", 0)
    range_high = metrics.get("range_high", 0)
    range_low = metrics.get("range_low", 0)
    adx = metrics.get("adx", 0)
    pdi = metrics.get("plus_di", 0)
    mdi = metrics.get("minus_di", 0)

    # Confirm breakout conditions
    bullish_breakout = (current_price > range_high and pdi >
                        mdi and adx > ADX_BREAKOUT_THRESHOLD)
    bearish_breakout = (current_price < range_low and mdi >
                        pdi and adx > ADX_BREAKOUT_THRESHOLD)

    if not bullish_breakout and not bearish_breakout:
        print(f"  [S3] No confirmed breakout for {pair}. "
              f"Price {current_price} within range {range_low}–{range_high}. HOLD.")
        return None

    confirmed_action = "BUY" if bullish_breakout else "SELL"
    print(f"  [S3] {confirmed_action} breakout confirmed for {pair}.")

    try:
        sl, tp, _ = estimate_sl_tp(pair, confirmed_action)
    except Exception as e:
        print(f"  [S3] ATR failed: {e}")
        return None

    return TradeSignal(
        pair_to_trade=pair,
        action=confirmed_action,
        confidence_score=0.75,
        stop_loss=sl,
        take_profit=tp,
        reasoning=(f"[BREAKOUT] {pair} closed {'above' if bullish_breakout else 'below'} "
                   f"range {'high' if bullish_breakout else 'low'} "
                   f"({range_high if bullish_breakout else range_low}). "
                   f"ADX={adx:.1f}, +DI={pdi:.1f}, -DI={mdi:.1f}.")
    )


# ==========================================
# STRATEGY 4: PULLBACK CONFIRM
# ==========================================

def place_pullback_limit_order(pair, action, units, entry, sl, tp, expiry_minutes=60):
    order_units = units if action == "BUY" else -units

    gtd_time = (
        datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    payload = {
        "order": {
            "instrument": pair,
            "units": str(order_units),
            "price": format_price_for_instrument(entry, pair),
            "type": "LIMIT",
            "timeInForce": "GTD",
            "gtdTime": gtd_time,
            "positionFill": "DEFAULT",
            "stopLossOnFill": {
                "price": format_price_for_instrument(sl, pair)
            },
            "takeProfitOnFill": {
                "price": format_price_for_instrument(tp, pair)
            },
            "clientExtensions": {
                "comment": f"S4 pullback {action} limit",
                "tag": "s4-pullback"
            }
        }
    }

    req = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=payload)
    oanda_client.request(req)

    order_id = req.response.get("orderCreateTransaction", {}).get("id")
    print(
        f"  [S4] Pullback LIMIT placed: {action} {pair} @ {entry} "
        f"id={order_id} expires={gtd_time}"
    )


# ==========================================
# MAIN CYCLE
# ==========================================
def run_cycle():
    profile = RISK_PROFILE[RISK_LEVEL]
    print(f"\n[{datetime.now().isoformat()}] === Scheduled check | "
          f"Meta: {META_MODE} | Risk: {RISK_LEVEL} ===")

    try:
        # 1. Currency strength — find best pair candidate
        pair, action, rationale, scores = with_retry(
            analyze_currency_strength, max_attempts=3, delay=5, label="currency_strength"
        )

        if FORCE_TEST_PAIR:
            pair = TEST_PAIR
            print(f"  [TEST] Forcing test pair: {pair}")

        if not pair and action != "COILING":
            print("  [CYCLE] No pair candidate from strength matrix. Skipping.")
            return

        # COILING state: currencies compressed — skip regime/strategy, go straight to S4
        if action == "COILING":
            print(
                "\n  [CYCLE] COILING detected. Routing directly to S4 breakout stop orders.")
            print(f"  Rationale: {rationale}")
            if pair:
                existing = get_open_position(pair)
                long_u = float((existing or {}).get(
                    "long",  {}).get("units", 0))
                short_u = float((existing or {}).get(
                    "short", {}).get("units", 0))
                if long_u != 0 or short_u != 0:
                    print(
                        f"  [S4] Open position exists for {pair}. Skipping stop placement.")
                else:
                    from breaking_entry import place_breakout_stop_orders
                    order_ids = place_breakout_stop_orders(
                        pair, profile["units"],
                        duration_hours=BREAKOUT_DURATION_HOURS,
                        width_pct=BREAKOUT_WIDTH_PCT / 100  # convert % to ratio
                    )
                    print(f"  [S4] Placed {len(order_ids)} stop orders." if order_ids
                          else "  [S4] No stop orders placed (range too wide for S4 criteria).")
            return

        # STRONG state: skip LLM gate, enter market order immediately
        force_immediate = (scores[max(scores, key=scores.get)] -
                           scores[min(scores, key=scores.get)]) > 1.5
        if force_immediate:
            print(
                "\n  [CYCLE] STRONG TREND detected. Entering immediately without LLM gate.")

        # 2. Regime detection on that pair
        print(f"\n[2/4] Regime detection on {pair}...")
        try:
            regime, metrics = with_retry(
                detect_regime, pair, max_attempts=3, delay=5, label="regime_detector"
            )

            if FORCE_TEST_PAIR:
                pdi = metrics.get("plus_di", 0)
                mdi = metrics.get("minus_di", 0)
                action = "BUY" if pdi > mdi else "SELL"
                print(f"  [TEST] Forcing test action from regime: {action}")

        except Exception as e:
            print(f"  [CYCLE] Regime detection failed after all retries: {e}")
            print(f"  [CYCLE] Skipping cycle — cannot trade without regime data.")
            return

        # 3. Meta selector — pick strategy
        print(f"\n[3/4] Meta selector ({META_MODE})...")
        if META_MODE == "MANUAL":
            strategy = MANUAL_STRATEGY
            print(
                f"  [MANUAL] Using hardcoded strategy {strategy}: {STRATEGY_LABELS[strategy]}")
        elif META_MODE == "RULES_SELECTS":
            strategy = select_strategy_by_rules(regime)
            print(
                f"  [RULES] Regime={regime} → strategy {strategy}: {STRATEGY_LABELS[strategy]}")
        elif META_MODE == "AI_SELECTS":
            sentiment_for_meta = get_latest_news_sentiment()
            strength_context = format_strength_for_llm(
                scores, pair, action, rationale)
            strategy = select_strategy_by_ai(
                pair, regime, metrics, strength_context, sentiment_for_meta)
        else:
            print(f"  [ERROR] Unknown META_MODE: {META_MODE}")
            return

        # 4. Run chosen strategy
        print(f"\n[4/4] Running {STRATEGY_LABELS[strategy]}...")
        if USE_GEMINI_AI and quota_guard.is_available():
            try:
                sentiment = with_retry(
                    get_latest_news_sentiment, max_attempts=3, delay=10, label="news_sentiment"
                )
                quota_guard.record_call()
            except Exception as e:
                quota_guard.handle_error(e)
                sentiment = ("Quota exhausted — no live sentiment. "
                             "Rules-based signals only this cycle.")
        else:
            sentiment = ("Quota exhausted — no live sentiment. "
                         "Rules-based signals only this cycle.")

        signal = None

        if strategy == STRATEGY_TREND:
            signal = run_trend_combined(
                pair, action, rationale, scores, profile, sentiment,
                skip_llm=force_immediate
            )

        elif strategy == STRATEGY_RANGE:
            existing = get_open_position(pair)
            if existing:
                long_u = float(existing.get("long", {}).get("units", 0))
                short_u = float(existing.get("short", {}).get("units", 0))
                if long_u != 0 or short_u != 0:
                    print(f"  [S2] Open position exists for {pair}. Skipping.")
                    return

            run_range_reversion(pair, action, profile)
            return

        elif strategy == STRATEGY_BREAKOUT:
            signal = run_breakout_confirm(
                pair, action, metrics, profile, sentiment)

        elif strategy == STRATEGY_PULLBACK:
            try:
                existing = with_retry(
                    get_open_position, pair,
                    max_attempts=3, delay=5, label="open_position_check"
                )
            except Exception as e:
                print(
                    f"  [S4] Could not check open position: {e}. Skipping cycle.")
                return

            if existing:
                long_u = float(existing.get("long",  {}).get("units", 0))
                short_u = float(existing.get("short", {}).get("units", 0))
                if long_u != 0 or short_u != 0:
                    print(f"  [S4] Open position exists for {pair}. Skipping.")
                    return

            # Check for ANY existing S4 stop orders on this pair
            try:
                pending_bounce = with_retry(
                    has_pending_order, pair, tag="s4-bounce-stop",
                    max_attempts=3, delay=5, label="pending_bounce_check"
                )
                pending_break = with_retry(
                    has_pending_order, pair, tag="s4-break-stop",
                    max_attempts=3, delay=5, label="pending_break_check"
                )
            except Exception as e:
                print(
                    f"  [S4] Could not check pending orders: {e}. Skipping cycle.")
                return

            if pending_bounce or pending_break:
                print(
                    f"  [S4] Pending S4 stop orders already exist for {pair}. Skipping.")
                return

            # Get S/R levels
            levels = get_support_resistance(
                pair, granularity="H1", count=200, window=3)
            support = levels["support"]
            resistance = levels["resistance"]
            dp = 3 if pair.endswith("_JPY") else 5
            pip = 0.01 if pair.endswith("_JPY") else 0.0001
            range_size = resistance - support

            # SL: 20 pips below support (outside noise zone)
            # TP bounce: project range width above resistance
            # TP breakout: project 2× range width above resistance
            sl = round(support - pip * 20, dp)
            tp_bounce = round(resistance + range_size,       dp)
            tp_breakout = round(resistance + range_size * 2.0, dp)

            print(
                f"  [S4] {pair} | support={support} | resistance={resistance}")
            print(
                f"  [S4] SL={sl} | TP-bounce={tp_bounce} | TP-breakout={tp_breakout}")

            # Bounce stop: just above support (confirms pullback reversal)
            place_stop_entry_order(
                pair, action, profile["units"],
                entry=round(support + pip * 3, dp),
                sl=sl, tp=tp_bounce,
                expiry_minutes=EXPIRE_AFTER,
                label="S4-BOUNCE"
            )

            # Breakout stop: just above resistance (confirms momentum continuation)
            place_stop_entry_order(
                pair, action, profile["units"],
                entry=round(resistance + pip * 3, dp),
                sl=sl, tp=tp_breakout,
                expiry_minutes=EXPIRE_AFTER,
                label="S4-BREAK"
            )
            return

        # 5. Risk gate + execution (only for market orders from S1 and S3)
        if signal is None:
            print("  DECISION  : No signal / HOLD this cycle.")
            return

        print(f"\n  DECISION  : {signal.action} {signal.pair_to_trade}")
        print(f"  CONFIDENCE: {signal.confidence_score * 100:.1f}%")
        print(f"  SL / TP   : {signal.stop_loss} / {signal.take_profit}")
        print(f"  REASONING : {signal.reasoning}")

        if signal.action == "HOLD":
            print("[SCHEDULED] -> HOLD. No order placed.")
            return

        if signal.confidence_score < profile["min_confidence"]:
            print(f"[SCHEDULED] -> Confidence {signal.confidence_score:.0%} below "
                  f"threshold {profile['min_confidence']:.0%} for risk level {RISK_LEVEL}. Skipping.")
            return

        print(f"[SCHEDULED] -> Executing {signal.action} {signal.pair_to_trade} | "
              f"{profile['units']:,} units | {STRATEGY_LABELS[strategy]}")
        execute_market_trade(signal, units_override=profile["units"])

    except Exception as e:
        import traceback
        print(f"[SCHEDULED ERROR] Cycle failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("=== Strategy Runner v2 ===")
    print(f"  Meta mode : {META_MODE}")
    print(f"  Risk level: {RISK_LEVEL}/10")
    print(f"  Interval  : every {CHECK_INTERVAL_MINUTES} min")
    print(f"  ATR source: {ATR_GRANULARITY} candles")
    print("  Press Ctrl+C to stop\n")

    run_cycle()
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(run_cycle)
    while True:
        schedule.run_pending()
        time.sleep(5)
