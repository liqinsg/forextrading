import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

META_MODE = "AI_SELECTS"    # "AI_SELECTS" | "RULES_SELECTS" | "MANUAL"
MANUAL_STRATEGY = 3         # only used when META_MODE = "MANUAL"


def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


OANDA_ENV = env_str("OANDA_ENV", "practice")
OANDA_API_TOKEN = env_str("OANDA_API_TOKEN", "")
OANDA_ACCOUNT_ID = env_str("OANDA_ACCOUNT_ID", "")

DECISION_MODE = env_int("DECISION_MODE", 3)
RISK_LEVEL = env_int("RISK_LEVEL", 10)
CHECK_INTERVAL_MINUTES = env_int("CHECK_INTERVAL_MINUTES", 15)
RANGE_LOOKBACK = env_int("RANGE_LOOKBACK", 20)

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
