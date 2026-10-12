"""Hard rules: a placement that breaks any of these is 'red' and is never recommended."""

from stow_recommender.config import MAX_WEIGHT_KG_BY_LEVEL, TINY_ALLOWED_BIN_TYPES
from stow_recommender.domain import Bin, Item


def fits(item: Item, bin_: Bin) -> bool:
    """The item fits if each side fits (in the best rotation) and there is enough free space."""
    sides_fit = all(i <= b for i, b in zip(item.dims_sorted, bin_.dims_sorted))
    return sides_fit and item.volume <= bin_.free_volume


def violations(item: Item, bin_: Bin) -> list[str]:
    """Every rule this placement breaks with reason."""
    reasons = []

    if not fits(item, bin_):
        reasons.append("does not fit")

    max_weight = MAX_WEIGHT_KG_BY_LEVEL[bin_.level]
    if item.weight_kg > max_weight:
        reasons.append(f"too heavy for level {bin_.level} (max {max_weight} kg)")

    if item.hazardous and bin_.has_hazardous:
        reasons.append("bin already holds a hazardous item")

    if item.is_tiny and bin_.bin_type not in TINY_ALLOWED_BIN_TYPES:
        reasons.append("tiny item needs a small bin")

    return reasons


def is_allowed(item: Item, bin_: Bin) -> bool:
    return not violations(item, bin_)
