"""Synthetic warehouse data.

"""

import math
import zlib

import numpy as np
import pandas as pd

from stow_recommender.config import POD_LAYOUT
from stow_recommender.domain import Item, build_pod
from stow_recommender.features import placement_features
from stow_recommender.rules import is_allowed

# The "comfortable fit" from the rules: not too empty, not packed tight.
FILL_TARGET = 0.65         # ideal share of the bin used after stowing
FILL_TOLERANCE = 0.18      # how quickly quality drops away from the target
COMFORT_CLEARANCE_CM = 2   # less free space than this around the item = hard to pick

# Item size groups: (share of items, min side cm, max side cm)
ITEM_GROUPS = {
    "tiny": (0.08, 0.8, 2.9),
    "small": (0.40, 4, 15),
    "medium": (0.35, 10, 28),
    "large": (0.17, 20, 42),
}
HAZARDOUS_SHARE = 0.15


def true_probability(fill_after: float, min_clearance_cm: float) -> float:
    """The hidden truth: chance a placement works out well.

    Best when the bin ends up comfortably full (around FILL_TARGET) and there is
    some space around the item. Too empty wastes space; too tight is hard to pick.
    """
    fill_quality = math.exp(-(((fill_after - FILL_TARGET) / FILL_TOLERANCE) ** 2))
    clearance_quality = 1 / (1 + math.exp(-(min_clearance_cm - COMFORT_CLEARANCE_CM) / 0.7))
    return 0.05 + 0.9 * fill_quality * clearance_quality


def hidden_factor(item_id: str, bin_id: str) -> float:
    """Things the data doesn't capture, e.g. how the items already in a bin are arranged.

    The model can't see this, so it can not rank perfectly.
    Same value every time for the same (item, bin), between 0.75 and 1.25 (average 1).
    """
    u = zlib.crc32(f"{item_id}|{bin_id}".encode()) / 2**32
    return 0.75 + 0.5 * u


def true_good_stow_probability(item: Item, bin_) -> float:
    f = placement_features(item, bin_)
    p = true_probability(f["fill_after"], f["min_clearance_cm"]) * hidden_factor(item.item_id, bin_.bin_id)
    return min(p, 0.99)


def random_item(rng: np.random.Generator, item_id: str) -> Item:
    groups = list(ITEM_GROUPS)
    group = rng.choice(groups, p=[ITEM_GROUPS[g][0] for g in groups])
    _, lo, hi = ITEM_GROUPS[group]
    w, h, d = rng.uniform(lo, hi, size=3).round(1)
    density = rng.uniform(0.05, 0.6)  # g per cm³
    weight_kg = float(np.clip(w * h * d * density / 1000, 0.01, 30))
    return Item(item_id, float(w), float(h), float(d), round(weight_kg, 2),
                hazardous=bool(rng.random() < HAZARDOUS_SHARE))


def random_pod(rng: np.random.Generator, pod_id: str):
    """A pod whose bins are already partly filled, some with hazardous items."""
    fill_ratios, hazardous = {}, set()
    for level, bin_types in POD_LAYOUT.items():
        for column in range(1, len(bin_types) + 1):
            bin_id = f"{pod_id}-L{level}-C{column}"
            fill = float(rng.uniform(0, 0.85))
            fill_ratios[bin_id] = fill
            if fill > 0.05 and rng.random() < 0.15:
                hazardous.add(bin_id)
    return build_pod(pod_id, fill_ratios, hazardous)


def generate_history(n_events: int, seed: int) -> pd.DataFrame:
    """Simulated log of past stows: features of the chosen bin + whether it went well.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for event in range(n_events):
        item = random_item(rng, f"I{event}")
        pod = random_pod(rng, f"P{event}")
        allowed = [b for b in pod if is_allowed(item, b)]
        if not allowed:
            continue
        chosen = allowed[rng.integers(len(allowed))]
        good = rng.random() < true_good_stow_probability(item, chosen)
        rows.append({**placement_features(item, chosen), "good_stow": int(good), "event_id": event})
    return pd.DataFrame(rows)


def generate_scenarios(n: int, seed: int, min_allowed: int = 2) -> list:
    """(item, pod) pairs for evaluating rankings. Each has at least `min_allowed` options."""
    rng = np.random.default_rng(seed)
    scenarios = []
    while len(scenarios) < n:
        k = len(scenarios)
        item, pod = random_item(rng, f"S{k}"), random_pod(rng, f"SP{k}")
        if sum(is_allowed(item, b) for b in pod) >= min_allowed:
            scenarios.append((item, pod))
    return scenarios
