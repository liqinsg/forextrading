# custom_strategy.py
import os
from abc import ABC, abstractmethod
from ml_confirmation import ml_filter, ENABLE_ML_WEIGHTED_DOMINANCE

from utils import get_support_resistance, oanda_client
import config as _config
from config import (
    TRADE_PAIRS, SL_BUFFER_PIPS, SPREAD_PIPS,
    MIN_QUALIFYING_PAIRS, MIN_DOMINANCE_RATIO,
    ENABLE_VOLATILITY_NORMALIZED_DOMINANCE,
    ENABLE_ATR_SLTP, ENABLE_NEWS_FILTER, ENABLE_EMA_TREND,
    ENABLE_ATR_NORMALIZED_STRENGTH, ENABLE_STRENGTH_ACCELERATION,
    STRENGTH_ACCELERATION_WEIGHT, ENABLE_BREAKOUT_CONFIRMATION,
    BREAKOUT_CONFIRMATION_CLOSES,
    # NEW: Gemini configs
    GEMINI_API_KEY, GEMINI_NEWS_MODEL, GEMINI_NEWS_FALLBACK_MODEL,
    NEWS_LOG_PATH, NEWS_CURRENCIES
)

from utils.strategy_helpers import (
    get_candles, _atr_from_candles, get_atr_with_volatility_context,
    get_dominance_normalizer, get_pair_momentum, build_strength_matrix,
    format_strength_ranking, get_ma5_position, _ema, get_ema_trend_position,
    get_trend_position, check_ma5_alignment, get_previous_day_low,
    get_previous_day_high, confirmed_breakout, get_live_prices,
    NewsFilter
)

OANDA_ACCOUNT_ID = getattr(_config, "OANDA_ACCOUNT_ID", None) or os.getenv("OANDA_ACCOUNT_ID")
if not OANDA_ACCOUNT_ID:
    print("[STRATEGY] WARNING: OANDA_ACCOUNT_ID not found in config.py or environment.")

JPY_TRADE_PAIRS = [p for p in TRADE_PAIRS if p.endswith("_JPY")]
# _dropped = [p for p in TRADE_PAIRS if not p.endswith("_JPY")]
# if _dropped:
if _dropped := [p for p in TRADE_PAIRS if not p.endswith("_JPY")]:
    print(f"[STRATEGY] WARNING: non-JPY pairs found in TRADE_PAIRS and will be IGNORED: {_dropped}")

_news_filter = NewsFilter()


# ==========================================
# STRATEGY INTERFACE
# ==========================================
class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, scores: dict) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def rules_description(self) -> str:
        raise NotImplementedError


# ==========================================
# JPY TREND STRATEGY
# ==========================================
class JPYTrendStrategy(Strategy):
    JPY_PIP = _config.JPY_PIP
    MIN_MARKET_STRENGTH = _config.MIN_MARKET_STRENGTH
    FRONT_RUN_PIPS = _config.FRONT_RUN_PIPS
    MACRO_PROTECTION_PIPS = _config.MACRO_PROTECTION_PIPS
    MIN_RR = _config.MIN_RR

    ATR_PERIOD = _config.JPY_ATR_PERIOD
    ATR_HISTORY_LOOKBACK = _config.JPY_ATR_HISTORY_LOOKBACK
    ATR_SL_MULTIPLIER_NORMAL = _config.JPY_ATR_SL_MULTIPLIER_NORMAL
    ATR_SL_MULTIPLIER_HIGH_VOL = _config.JPY_ATR_SL_MULTIPLIER_HIGH_VOL
    ATR_SL_MULTIPLIER_LOW_VOL = _config.JPY_ATR_SL_MULTIPLIER_LOW_VOL
    ATR_RR_MULTIPLE = _config.JPY_ATR_RR_MULTIPLE

    def __init__(self, trade_pairs: list[str] | None = None):
        self.trade_pairs = trade_pairs if trade_pairs is not None else JPY_TRADE_PAIRS
        if not self.trade_pairs:
            print("[STRATEGY] WARNING: JPYTrendStrategy has no trade pairs configured.")

    def jpy_strength_rank(self, scores: dict) -> dict[str, float]:
        jpy_score = scores.get("JPY", 0.0)
        return {
            pair: scores.get(pair.split("_")[0], 0.0) - jpy_score
            for pair in self.trade_pairs
        }

    def generate_signals(self, scores: dict) -> list[dict]:
        _news_filter.reset_cycle()
        jpy_ranks = self.jpy_strength_rank(scores)

        max_gap = max(abs(v) for v in jpy_ranks.values()) if jpy_ranks else 0.0
        if max_gap < self.MIN_MARKET_STRENGTH:
            print(f"  [STRATEGY] Global JPY gap ({max_gap:.4f}) below minimum floor.")
            max_gap = self.MIN_MARKET_STRENGTH

        ranked_pairs = sorted(jpy_ranks.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  JPY cross priority: {' > '.join(f'{p}({s:+.3f})' for p, s in ranked_pairs)}")
        print("\n[STRATEGY] Step 2 — MA5 alignment check...")
        signals = []

        for pair, strength_score in ranked_pairs:
            print(f"\n  [{pair}] (strength score: {strength_score:+.4f})")

            dynamic_cutoff = max_gap * 0.4
            if abs(strength_score) < dynamic_cutoff:
                print(f"    → Skip: strength gap {abs(strength_score):.4f} < {dynamic_cutoff:.4f}")
                continue

            should_avoid, news_reason = _news_filter.should_avoid_pair(pair)
            if should_avoid:
                print(f"    → Skip: news risk - {news_reason}")
                continue

            direction = check_ma5_alignment(pair)
            if direction is None:
                print("    → No alignment (Mixed timeframes)")
                continue

            if direction == "BUY" and strength_score < 0:
                print("    → Mixed Signal: Techs BUY but Strength favors JPY. Skipping.")
                continue
            if direction == "SELL" and strength_score > 0:
                print("    → Mixed Signal: Techs SELL but Strength favors Base. Skipping.")
                continue
            should_avoid_ml, ml_reason = ml_filter.should_avoid_pair(pair, direction)
            if should_avoid_ml:
                print(f"    → Skip: ML confidence - {ml_reason}")
                continue

            prices = get_live_prices(pair)
            if prices is None:
                print("    → Cannot get live prices. Skipping.")
                continue

            daily_levels = get_support_resistance(pair, granularity="D", count=60, window=3)
            weekly_levels = get_support_resistance(pair, granularity="W", count=52, window=2)

            if (daily_levels["support"] is None or daily_levels["resistance"] is None or
                    weekly_levels["support"] is None or weekly_levels["resistance"] is None):
                print(f"    → Structural tracking error. Skipping {pair}.")
                continue

            print(f"    [D1] S={daily_levels['support']:.3f} | R={daily_levels['resistance']:.3f}")
            print(f"    [W1] S={weekly_levels['support']:.3f} | R={weekly_levels['resistance']:.3f}")

            if ENABLE_ATR_SLTP:
                atr, z_score = get_atr_with_volatility_context(pair, self.ATR_PERIOD, self.ATR_HISTORY_LOOKBACK)
                if atr is None or atr <= 0:
                    print("    → Skip: ATR unavailable.")
                    continue

                if z_score is None:
                    sl_multiplier = self.ATR_SL_MULTIPLIER_NORMAL
                    vol_label = "volatility unknown"
                elif z_score > 1:
                    sl_multiplier = self.ATR_SL_MULTIPLIER_HIGH_VOL
                    vol_label = f"HIGH volatility (z={z_score:+.2f})"
                elif z_score < -1:
                    sl_multiplier = self.ATR_SL_MULTIPLIER_LOW_VOL
                    vol_label = f"LOW volatility (z={z_score:+.2f})"
                else:
                    sl_multiplier = self.ATR_SL_MULTIPLIER_NORMAL
                    vol_label = f"NORMAL volatility (z={z_score:+.2f})"

                sl_distance = atr * sl_multiplier
                tp_distance = sl_distance * self.ATR_RR_MULTIPLE

                print(f"    [ATR] {atr:.3f} | {vol_label} | SL={sl_distance:.3f} | TP={tp_distance:.3f}")

                if direction == "BUY":
                    entry = prices["ask"]
                    sl = round(entry - sl_distance, 3)
                    tp = round(entry + tp_distance, 3)
                else:
                    entry = prices["bid"]
                    sl = round(entry + sl_distance, 3)
                    tp = round(entry - tp_distance, 3)

                sl_reference = f"ATR x{sl_multiplier}"
                target_type = f"ATR x{sl_multiplier * self.ATR_RR_MULTIPLE:.2f}"

                if direction == "BUY" and entry > (weekly_levels["resistance"] - self.MACRO_PROTECTION_PIPS * self.JPY_PIP):
                    print("    → Abort BUY: too close to Weekly Resistance")
                    continue
                if direction == "SELL" and entry < (weekly_levels["support"] + self.MACRO_PROTECTION_PIPS * self.JPY_PIP):
                    print("    → Abort SELL: too close to Weekly Support")
                    continue

            elif direction == "BUY":
                entry = prices["ask"]
                sl_reference = "Daily Support"
                sl = round(daily_levels["support"] - (SL_BUFFER_PIPS + SPREAD_PIPS) * self.JPY_PIP, 3)

                broke_out = confirmed_breakout(pair, daily_levels["resistance"], "above") if ENABLE_BREAKOUT_CONFIRMATION else entry > daily_levels["resistance"]
                tp = round((weekly_levels["resistance"] if broke_out else daily_levels["resistance"]) - self.FRONT_RUN_PIPS * self.JPY_PIP, 3)
                target_type = "Weekly Resistance" if broke_out else "Daily Resistance"

                if entry > weekly_levels["resistance"] - self.MACRO_PROTECTION_PIPS * self.JPY_PIP:
                    print("    → Abort BUY: too close to Weekly Resistance")
                    continue
                if tp <= entry or sl >= entry:
                    print("    → Skip BUY: Invalid SL/TP levels")
                    continue

            else:
                entry = prices["bid"]
                sl_reference = "Daily Resistance"
                sl = round(daily_levels["resistance"] + (SL_BUFFER_PIPS + SPREAD_PIPS) * self.JPY_PIP, 3)

                broke_down = confirmed_breakout(pair, daily_levels["support"], "below") if ENABLE_BREAKOUT_CONFIRMATION else entry < daily_levels["support"]
                tp = round((weekly_levels["support"] if broke_down else daily_levels["support"]) + self.FRONT_RUN_PIPS * self.JPY_PIP, 3)
                target_type = "Weekly Support" if broke_down else "Daily Support"

                if entry < weekly_levels["support"] + self.MACRO_PROTECTION_PIPS * self.JPY_PIP:
                    print("    → Abort SELL: too close to Weekly Support")
                    continue
                if tp >= entry or sl <= entry:
                    print("    → Skip SELL: Invalid SL/TP levels")
                    continue

            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr = reward / risk if risk > 0 else 0.0
            if rr < self.MIN_RR:
                print(f"    → Skip: R:R {rr:.2f} < {self.MIN_RR}")
                continue

            print(f"    → SIGNAL: {direction} {pair}")
            print(f"       Entry: {entry} | SL: {sl} ({sl_reference}) | TP: {tp} ({target_type}) | R:R {rr:.2f}")

            signals.append({
                "pair": pair,
                "action": direction,
                "entry": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "strength_score": strength_score,
                "risk_reward": round(rr, 2),
                "reasoning": f"Aligned {direction} | SL={sl_reference} | TP={target_type}"
            })

        buy_signals = [s for s in signals if s["action"] == "BUY"]
        sell_signals = [s for s in signals if s["action"] == "SELL"]

        def _dominance_weight(s: dict) -> float:
            raw = abs(s["strength_score"])
            if not ENABLE_VOLATILITY_NORMALIZED_DOMINANCE:
                pass
            else:
                atr = get_dominance_normalizer(s["pair"])
                raw = raw / atr if (atr and atr > 0) else raw
            if ENABLE_ML_WEIGHTED_DOMINANCE:
                conf = ml_filter.get_confidence(s["pair"], s["action"])
                raw *= conf
            return raw

            atr = get_dominance_normalizer(s["pair"])
            return raw / atr if (atr and atr > 0) else raw

        buy_weight = sum(_dominance_weight(s) for s in buy_signals)
        sell_weight = sum(_dominance_weight(s) for s in sell_signals)

        print(f"\n  [CORROBORATION] BUY={len(buy_signals)} (w={buy_weight:.4f}) | SELL={len(sell_signals)} (w={sell_weight:.4f})")

        if not buy_signals and not sell_signals:
            return []

        dominant, dom_label, dom_weight, opp_weight = (buy_signals, "BUY", buy_weight, sell_weight) if buy_weight >= sell_weight else (sell_signals, "SELL", sell_weight, buy_weight)

        if len(dominant) < MIN_QUALIFYING_PAIRS:
            print(f"  → Only {len(dominant)}/{MIN_QUALIFYING_PAIRS} pairs — HOLD")
            return []
        if opp_weight > 0 and dom_weight < opp_weight * MIN_DOMINANCE_RATIO:
            print(f"  → {dom_label} weight insufficient — HOLD")
            return []

        print(f"  ✅ {dom_label} consensus confirmed")
        return dominant

    def rules_description(self) -> str:
        news_status = "ON" if ENABLE_NEWS_FILTER else "OFF"
        trend_method = "EMA10/EMA20" if ENABLE_EMA_TREND else "MA5"
        strength_method = "ATR-normalized" if ENABLE_ATR_NORMALIZED_STRENGTH else "raw %"
        accel_status = f"ON (w={STRENGTH_ACCELERATION_WEIGHT})" if ENABLE_STRENGTH_ACCELERATION else "OFF"
        breakout_status = f"ON ({BREAKOUT_CONFIRMATION_CLOSES} closes)" if ENABLE_BREAKOUT_CONFIRMATION else "OFF"
        sltp_method = "ATR-based" if ENABLE_ATR_SLTP else "Structural S/R"
        dominance = "ATR-normalized" if ENABLE_VOLATILITY_NORMALIZED_DOMINANCE else "raw"

        text = f"""
RULES:
  • Strength: {strength_method}, Acceleration: {accel_status}
  • Trend filter: {trend_method}
  • SL/TP: {sltp_method}
  • Breakout confirmation: {breakout_status}
  • News filter: {news_status}
  • Min R:R: {self.MIN_RR}
  • Consensus: ≥{MIN_QUALIFYING_PAIRS} pairs, ratio ≥{MIN_DOMINANCE_RATIO}
  • Dominance weighting: {dominance}
"""
        if ENABLE_NEWS_FILTER and _news_filter.degraded():
            text += "\n⚠️ NEWS FILTER FAILED THIS CYCLE\n"
        return text


# ==========================================
# RUNNER & ENTRY POINT
# ==========================================
def run_strategy(strategy: Strategy) -> tuple[str, dict | None]:
    print("[STRATEGY] Step 1 — Currency strength ranking...")
    scores = build_strength_matrix()
    strength_report = format_strength_ranking(scores)
    print(strength_report)

    signals = strategy.generate_signals(scores)
    best = max(signals, key=lambda s: abs(s["strength_score"])) if signals else None

    report = "=== JPY TREND STRATEGY REPORT ===\n\n"
    report += strength_report + "\n\n=== SIGNALS ===\n"
    if signals:
        for s in signals:
            report += f"• {s['action']} {s['pair']} | SL={s['stop_loss']} TP={s['take_profit']} R:R={s['risk_reward']}\n"
        report += f"\nBest: {best['action']} {best['pair']}\nReason: {best['reasoning']}\n"
    else:
        report += "• No valid signals — HOLD\n"

    report += strategy.rules_description()
    return report, best


_active_strategy = JPYTrendStrategy()


def analyze_custom_strategy() -> str:
    report, best = run_strategy(_active_strategy)
    analyze_custom_strategy._last_signal = best
    return report


analyze_custom_strategy._last_signal = None


def get_last_signal() -> dict | None:
    return analyze_custom_strategy._last_signal