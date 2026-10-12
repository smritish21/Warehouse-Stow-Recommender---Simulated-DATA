"""Pod layout, hard rules and settings, all in one place."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42

# ---------- Pod layout ----------
# A pod is a shelving unit: 6 levels (1 = floor, 6 = top shelf), 4 bins per level.
# Bin inner sizes in cm: (width, height, depth)
BIN_SIZES = {
    "small": (25, 20, 35),
    "medium": (35, 30, 40),
    "large": (50, 40, 45),
}

POD_LAYOUT = {
    1: ["large", "large", "large", "large"],
    2: ["large", "large", "large", "large"],
    3: ["medium", "medium", "medium", "medium"],
    4: ["medium", "small", "small", "medium"],
    5: ["small", "small", "small", "small"],
    6: ["small", "small", "small", "small"],
}

# ---------- Hard rules (never broken, whatever the model says) ----------

# Rule: heavy items stay low. Maximum item weight allowed on each level.
MAX_WEIGHT_KG_BY_LEVEL = {1: 25, 2: 25, 3: 12, 4: 8, 5: 4, 6: 4}

# Rule: hazardous items (can break or spill) never share a bin with another hazardous item.
# (Implemented in rules.py; no setting needed.)

# Rule: tiny items (longest side under 3 cm, about an inch) only go in small bins,
# where they are easy to find and can't get lost or fall out.
TINY_MAX_DIM_CM = 3
TINY_ALLOWED_BIN_TYPES = {"small"}

# ---------- Recommendation output ----------
BEST_FIT_COUNT = 3  # top N allowed bins are shown as "best fit"
