
"""
ml_confirmation.py
===================
Optional ML confidence layer for JPYTrendStrategy signals.

Does NOT generate trade signals on its own. Sits alongside the existing
rule-based pipeline (MA5 alignment, ATR SL/TP, currency strength dominance)
as an additional confirmation gate + confidence score — same switch-gated
pattern as NewsFilter / ENABLE_ATR_SLTP / etc. in custom_strategy.py.

Two integration points:
  1. should_avoid_pair(pair, direction) -> gate a signal out entirely
     (drop into custom_strategy.py's generate_signals(), next to the
     existing _news_filter.should_avoid_pair() call)
  2. get_confidence(pair, direction)    -> weight dominance, or replace
     the hardcoded confidence_score=0.85 in scheduled_runner.py

Design choices made to avoid the pitfalls found in forex_strategy.py:
  - Trains on OANDA candles via utils.strategy_helpers.get_candles(), NOT
    yfinance, so train-time features match live-time features exactly.
  - Holdout backtest (not in-sample) so confidence numbers reflect
    genuine out-of-sample skill rather than memorized training data.
  - No volume-derived features. OANDA FX candles have no real traded
    volume (tick-count proxy at best); the same near-zero-division
    blowup risk identified in forex_strategy.py's VolChange applies here,
    so those features are dropped entirely.
  - Model is retrained on a TTL (default: once per day) and cached
    per-pair in memory, not on every 15-min cycle.

NOT yet wired into custom_strategy.py or scheduled_runner.py — this is
a standalone draft. See the two call sites above for how to plug it in
once you're happy with the holdout numbers it logs on first training.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import talib
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from utils.strategy_helpers import get_candles

import config as _config

# ----------------------
# Config — add these to config.py; safe defaults if you don't
# ----------------------
ENABLE_ML_CONFIRMATION = getattr(_config, "ENABLE_ML_CONFIRMATION", False)
ENABLE_ML_WEIGHTED_DOMINANCE = getattr(_config, "ENABLE_ML_WEIGHTED_DOMINANCE", False)
ML_MIN_CONFIDENCE = getattr(_config, "ML_MIN_CONFIDENCE", 0.55)
ML_RETRAIN_HOURS = getattr(_config, "ML_RETRAIN_HOURS", 24)
ML_TRAIN_GRANULARITY = getattr(_config, "ML_TRAIN_GRANULARITY", "H1")
ML_TRAIN_CANDLE_COUNT = getattr(_config, "ML_TRAIN_CANDLE_COUNT", 3000)
ML_HOLDOUT_FRACTION = getattr(_config, "ML_HOLDOUT_FRACTION", 0.2)
ML_LABEL_HORIZON = getattr(_config, "ML_LABEL_HORIZON", 3)  # bars ahead for future_return
ML_MIN_HOLDOUT_F1 = getattr(_config, "ML_MIN_HOLDOUT_F1", 0.0)  # floor below which model is distrusted
ML_BACKTEST_FEE_PCT = getattr(_config, "ML_BACKTEST_FEE_PCT", 0.0002)  # round-trip cost per position change, as a fraction (0.0002 = 2 pips on a ~1.0-quoted pair equivalent)


class _PairModel:
    """Holds a trained pipeline + last-fit metadata for a single pair."""

    def __init__(self):
        self.pipeline = None
        self.trained_at = None
        self.holdout_accuracy = None
        self.holdout_f1 = None

    def is_stale(self) -> bool:
        if self.pipeline is None or self.trained_at is None:
            return True
        return datetime.now() - self.trained_at > timedelta(hours=ML_RETRAIN_HOURS)

    def is_trustworthy(self) -> bool:
        return self.pipeline is not None and (self.holdout_f1 or 0.0) >= ML_MIN_HOLDOUT_F1


def _candles_to_df(candles: list) -> pd.DataFrame:
    """
    Converts OANDA's raw candle format — a list of dicts like
    {'complete': True, 'volume': N, 'time': '...Z', 'mid': {'o','h','l','c'}}
    — into a DataFrame indexed by time with float open/high/low/close
    columns. Drops incomplete (still-forming) candles, since including
    a partial bar would let the model "see" a return that hasn't
    finished happening yet.
    """
    rows = []
    for c in candles:
        if not c.get("complete", True):
            continue
        mid = c.get("mid") or {}
        try:
            rows.append({
                "time": c["time"],
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def _build_features(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Candle-derived features only — deliberately no volume. Expects an
    OHLC DataFrame already converted by _candles_to_df().
    """
    df = candles.copy()
    close = np.asarray(df["close"], dtype=float)
    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)

    feats = pd.DataFrame(index=df.index)
    feats["MA20"] = talib.SMA(close, timeperiod=20)
    feats["ATR"] = talib.ATR(high, low, close, timeperiod=14)
    feats["RSI"] = talib.RSI(close, timeperiod=14)
    feats["CCI"] = talib.CCI(high, low, close, timeperiod=20)
    feats["PriceChange"] = np.append(np.nan, np.diff(close) / close[:-1])

    for lag in (1, 2, 3):
        feats[f"RSI_lag{lag}"] = feats["RSI"].shift(lag)
        feats[f"PriceChange_lag{lag}"] = feats["PriceChange"].shift(lag)

    return feats


def _build_labels(candles: pd.DataFrame, horizon: int) -> pd.Series:
    close = np.asarray(candles["close"], dtype=float)
    future_return = pd.Series(close, index=candles.index).pct_change(horizon).shift(-horizon)
    return (future_return > 0).astype(int)


def _fit_pipeline(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, float, float]:
    """
    Fit on the first (1 - ML_HOLDOUT_FRACTION) of the series in time
    order, score on the held-out tail. This is a simple chronological
    holdout rather than in-sample scoring — the whole point is that
    holdout_accuracy/holdout_f1 reflect data the model never trained on.
    """
    split_idx = int(len(X) * (1 - ML_HOLDOUT_FRACTION))
    X_train, X_hold = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_hold = y.iloc[:split_idx], y.iloc[split_idx:]

    base = SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=42)
    calibrated = CalibratedClassifierCV(base, ensemble=False)
    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", calibrated)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_hold)
    acc = accuracy_score(y_hold, preds)
    f1 = f1_score(y_hold, preds, zero_division=0)
    return pipeline, acc, f1


class MLConfirmationFilter:
    """
    Lazy, per-pair-cached ML confidence layer.

    should_avoid_pair(pair, direction) -> (bool, reason)
    get_confidence(pair, direction)    -> float in [0.0, 1.0]

    Both are no-ops (never avoid / always neutral 0.5) when
    ENABLE_ML_CONFIRMATION is False, so importing and calling this is
    safe even before you've validated the models — mirrors how
    NewsFilter behaves when ENABLE_NEWS_FILTER is off.
    """

    def __init__(self):
        self._models: dict[str, _PairModel] = {}

    def _get_or_train(self, pair: str) -> _PairModel:
        model = self._models.setdefault(pair, _PairModel())
        if not model.is_stale():
            return model

        print(f"  [ML] Training/refreshing model for {pair} ({ML_TRAIN_GRANULARITY}, "
              f"{ML_TRAIN_CANDLE_COUNT} candles)...")
        try:
            raw = get_candles(pair, granularity=ML_TRAIN_GRANULARITY, count=ML_TRAIN_CANDLE_COUNT)
            candles = _candles_to_df(raw) if raw is not None else None
            if candles is None or len(candles) < 200:
                print(f"  [ML] {pair}: insufficient candle history, skipping training.")
                return model

            X = _build_features(candles)
            y = _build_labels(candles, ML_LABEL_HORIZON)
            data = pd.concat([X, y.rename("label")], axis=1).dropna()
            if len(data) < 100:
                print(f"  [ML] {pair}: insufficient rows after feature/label cleanup.")
                return model

            Xc, yc = data.drop(columns=["label"]), data["label"]
            pipeline, acc, f1 = _fit_pipeline(Xc, yc)

            model.pipeline = pipeline
            model.trained_at = datetime.now()
            model.holdout_accuracy = acc
            model.holdout_f1 = f1

            trust_flag = "" if model.is_trustworthy() else "  ⚠️ below ML_MIN_HOLDOUT_F1, treating as neutral"
            print(f"  [ML] {pair}: retrained. Holdout acc={acc:.2%} f1={f1:.2f}{trust_flag}")
        except Exception as e:
            print(f"  [ML] {pair}: training failed — {e}")

        return model

    def get_confidence(self, pair: str, direction: str) -> float:
        """
        Probability the model assigns to price rising (direction ==
        "BUY") or falling (direction == "SELL") over the next
        ML_LABEL_HORIZON bars. Returns 0.5 (neutral) if disabled,
        untrained, below the holdout F1 floor, or on any failure —
        never raises, so a broken model can't take down a live cycle.
        """
        if not ENABLE_ML_CONFIRMATION:
            return 0.5

        model = self._get_or_train(pair)
        if model.pipeline is None or not model.is_trustworthy():
            return 0.5

        try:
            raw = get_candles(pair, granularity=ML_TRAIN_GRANULARITY, count=100)
            candles = _candles_to_df(raw) if raw is not None else pd.DataFrame()
            X = _build_features(candles).dropna()
            if X.empty:
                return 0.5
            latest = X.tail(1)
            prob_up = float(model.pipeline.predict_proba(latest)[0, 1])
            return prob_up if direction == "BUY" else (1 - prob_up)
        except Exception as e:
            print(f"  [ML] {pair}: inference failed — {e}. Treating as neutral.")
            return 0.5

    def should_avoid_pair(self, pair: str, direction: str) -> tuple[bool, str]:
        """
        Gate a signal out if ML confidence is below ML_MIN_CONFIDENCE.
        Always returns (False, "") when ENABLE_ML_CONFIRMATION is off.
        """
        if not ENABLE_ML_CONFIRMATION:
            return False, ""

        conf = self.get_confidence(pair, direction)
        if conf < ML_MIN_CONFIDENCE:
            return True, f"ML confidence {conf:.2f} < {ML_MIN_CONFIDENCE}"
        return False, ""


# Module-level singleton, mirroring the _news_filter pattern in custom_strategy.py
ml_filter = MLConfirmationFilter()


# ==========================================
# STANDALONE BACKTEST / VALIDATION
# ==========================================
def run_backtest(pair: str) -> dict:
    """
    Trains on the first (1 - ML_HOLDOUT_FRACTION) of history only, then
    simulates trading purely on the held-out tail using predict_proba
    from that train-only model. No look-ahead: the model never sees
    the bars it's scored against here.

    Prints a human-readable report and returns the metrics dict.
    """
    print(f"\n=== ML Backtest: {pair} ===")
    raw = get_candles(pair, granularity=ML_TRAIN_GRANULARITY, count=ML_TRAIN_CANDLE_COUNT)
    candles = _candles_to_df(raw) if raw is not None else None
    if candles is None or len(candles) < 200:
        print(f"  Insufficient candle history for {pair}, skipping.")
        return {"pair": pair, "status": "insufficient_data"}

    X = _build_features(candles)
    y = _build_labels(candles, ML_LABEL_HORIZON)
    data = pd.concat([X, y.rename("label")], axis=1).dropna()
    if len(data) < 100:
        print(f"  Insufficient rows after feature/label cleanup for {pair}.")
        return {"pair": pair, "status": "insufficient_rows"}

    Xc, yc = data.drop(columns=["label"]), data["label"]
    pipeline, acc, f1 = _fit_pipeline(Xc, yc)

    split_idx = int(len(Xc) * (1 - ML_HOLDOUT_FRACTION))
    X_hold = Xc.iloc[split_idx:]

    proba = pipeline.predict_proba(X_hold)[:, 1]
    signal = np.where(proba > ML_MIN_CONFIDENCE, 1,
                       np.where(proba < (1 - ML_MIN_CONFIDENCE), -1, 0))
    signal = pd.Series(signal, index=X_hold.index)

    # Diagnostic: a model that just learned "mostly predict up" can post a
    # deceptively high F1 (scored on the positive class only) while barely
    # beating chance on accuracy. Compare what it predicts vs. what happened.
    predicted_up_rate = float((proba > 0.5).mean())
    actual_up_rate = float(yc.iloc[split_idx:].mean())

    price = candles["close"].reindex(X_hold.index)
    returns = price.pct_change().fillna(0)
    position = signal.shift(1).fillna(0)  # shift(1): trade on next bar's return, no look-ahead
    trade_flags = position.diff().abs().fillna(0)  # 1 or 2 on a position change, 0 otherwise

    gross_returns = position * returns
    fee_drag = trade_flags * ML_BACKTEST_FEE_PCT
    net_returns = gross_returns - fee_drag

    gross_equity = (1 + gross_returns).cumprod()
    net_equity = (1 + net_returns).cumprod()

    gross_return = gross_equity.iloc[-1] - 1
    net_return = net_equity.iloc[-1] - 1
    buy_hold_return = (price.iloc[-1] / price.iloc[0]) - 1
    drawdown = (net_equity / net_equity.cummax()) - 1
    max_dd = drawdown.min()
    num_trades = int((trade_flags > 0).sum())
    traded_returns = net_returns[net_returns != 0]
    win_rate = (traded_returns > 0).mean() if num_trades > 0 else 0.0

    metrics = {
        "pair": pair,
        "status": "ok",
        "holdout_bars": len(X_hold),
        "holdout_accuracy": acc,
        "holdout_f1": f1,
        "predicted_up_rate": predicted_up_rate,
        "actual_up_rate": actual_up_rate,
        "gross_return": gross_return,
        "net_return": net_return,
        "buy_hold_return": buy_hold_return,
        "max_drawdown": max_dd,
        "num_trades": num_trades,
        "win_rate": win_rate,
    }

    print(f"  Holdout bars       : {metrics['holdout_bars']}")
    print(f"  Holdout accuracy   : {acc:.2%}")
    print(f"  Holdout F1         : {f1:.2f}")
    print(f"  Predicted-up rate  : {predicted_up_rate:.1%}   (actual-up rate: {actual_up_rate:.1%})")
    print(f"  Gross return       : {gross_return:.2%}  (no fees)")
    print(f"  Net return         : {net_return:.2%}  (after {ML_BACKTEST_FEE_PCT:.4%} per position change, {num_trades} changes)")
    print(f"  Buy & hold return  : {buy_hold_return:.2%}")
    print(f"  Max drawdown       : {max_dd:.2%}  (net)")
    print(f"  Trades             : {num_trades}")
    print(f"  Win rate           : {win_rate:.2%}  (net, per bar held)")

    if abs(predicted_up_rate - 0.5) > 0.15 and abs(predicted_up_rate - actual_up_rate) > 0.1:
        print(f"  ⚠️  Predicted-up rate ({predicted_up_rate:.1%}) is skewed and diverges from "
              f"actual ({actual_up_rate:.1%}) — model may just be favoring one side rather than "
              f"discriminating; treat F1 with caution here.")

    return metrics


def main():
    """
    Standalone validation entry point:

        python ml_confirmation.py

    Trains + holdout-backtests every JPY pair in config.TRADE_PAIRS and
    prints a summary table. This does NOT touch the live/TTL-cached
    models used by MLConfirmationFilter at runtime — it's purely for
    deciding whether the approach has any edge before flipping
    ENABLE_ML_CONFIRMATION on in config.py.

    Numbers here are still simplified (no spread/slippage/fee modeling,
    single train/holdout split rather than walk-forward across multiple
    windows) — treat this as a sanity check, not a final validation.
    """
    pairs = [p for p in getattr(_config, "TRADE_PAIRS", []) if p.endswith("_JPY")]
    if not pairs:
        print("No JPY pairs found in config.TRADE_PAIRS.")
        return

    print("=== ML Confirmation — Standalone Backtest ===")
    print(f"Pairs           : {', '.join(pairs)}")
    print(f"Granularity     : {ML_TRAIN_GRANULARITY}")
    print(f"Candle count    : {ML_TRAIN_CANDLE_COUNT}")
    print(f"Holdout fraction: {ML_HOLDOUT_FRACTION}")
    print(f"Label horizon   : {ML_LABEL_HORIZON} bars")
    print(f"Confidence gate : {ML_MIN_CONFIDENCE}")

    results = [run_backtest(pair) for pair in pairs]

    print("\n=== SUMMARY ===")
    header = (f"{'Pair':<10}{'F1':>8}{'Acc':>8}{'PredUp':>8}"
              f"{'NetRet':>10}{'B&H':>10}{'MaxDD':>10}{'Trades':>8}{'Win%':>8}")
    print(header)
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['pair']:<10}  ({r['status']})")
            continue
        print(f"{r['pair']:<10}"
              f"{r['holdout_f1']:>8.2f}"
              f"{r['holdout_accuracy']:>8.1%}"
              f"{r['predicted_up_rate']:>8.1%}"
              f"{r['net_return']:>10.2%}"
              f"{r['buy_hold_return']:>10.2%}"
              f"{r['max_drawdown']:>10.2%}"
              f"{r['num_trades']:>8d}"
              f"{r['win_rate']:>8.1%}")

    trustworthy = [r for r in results if r.get("status") == "ok" and r["holdout_f1"] >= ML_MIN_HOLDOUT_F1]
    if not trustworthy:
        print(f"\n⚠️  No pair cleared ML_MIN_HOLDOUT_F1={ML_MIN_HOLDOUT_F1}. "
              f"Recommend leaving ENABLE_ML_CONFIRMATION off for now.")


if __name__ == "__main__":
    main()