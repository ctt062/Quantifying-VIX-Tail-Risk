"""Central configuration for the VIX shock study."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CACHE_FILE = RAW_DATA_DIR / "vix_history.parquet"
DEFAULT_TICKER = "^VIX"
DEFAULT_START = "2010-01-01"
DEFAULT_END = None  # Use latest available
BUSINESS_FREQ = "B"
WINSOR_ALPHA = 0.001
SHOCK_QUANTILES = (0.90, 0.95, 0.975)
PLOT_STYLE = "darkgrid"
RANDOM_STATE = 42
