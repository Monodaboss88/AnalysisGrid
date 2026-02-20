"""Configuration for the Polygon Signal Tool."""
import os

# Polygon.io API
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
POLYGON_BASE_URL = "https://api.polygon.io"

# Cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Analysis defaults
DEFAULT_LOOKBACK_DAYS = 365
HORIZONS = [1, 2, 3, 5]

# Severity buckets for drop classification
SEVERITY_BUCKETS = [
    ("Small (<0.5%)", 0, 0.005),
    ("Medium (0.5-1.5%)", 0.005, 0.015),
    ("Large (1.5-3%)", 0.015, 0.03),
    ("Severe (>3%)", 0.03, 1.0),
]

# Signal weights
WEIGHT_WIN_RATE = 0.40
WEIGHT_HIT_RATE = 0.30
WEIGHT_AVG_SCALP = 0.20
WEIGHT_SEVERITY = 0.10
