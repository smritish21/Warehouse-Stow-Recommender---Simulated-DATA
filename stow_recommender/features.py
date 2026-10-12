"""Turn an (item, bin) pair into numbers the model can learn from."""

import pandas as pd

from stow_recommender.domain import Bin, Item

FEATURE_NAMES = [
    "fill_before",       # how full the bin is now (0-1)
    "fill_after",        # how full it would be with this item (0-1)
    "min_clearance_cm",  # smallest gap between item and bin walls (best rotation)
    "item_volume",
    "bin_volume",
    "item_weight_kg",
    "level",
    "is_tiny",
]


def min_clearance(item: Item, bin_: Bin) -> float:
    """Smallest gap around the item. Near 0 means it barely fits and is hard to pick."""
    return min(b - i for b, i in zip(bin_.dims_sorted, item.dims_sorted))


def placement_features(item: Item, bin_: Bin) -> dict:
    return {
        "fill_before": bin_.fill_ratio,
        "fill_after": (bin_.used_volume + item.volume) / bin_.volume,
        "min_clearance_cm": min_clearance(item, bin_),
        "item_volume": item.volume,
        "bin_volume": bin_.volume,
        "item_weight_kg": item.weight_kg,
        "level": bin_.level,
        "is_tiny": int(item.is_tiny),
    }


def features_frame(item: Item, bins: list[Bin]) -> pd.DataFrame:
    """One row per candidate bin, columns in a fixed order."""
    return pd.DataFrame([placement_features(item, b) for b in bins], columns=FEATURE_NAMES)
