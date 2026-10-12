"""Shared fixtures (pytest loads this file automatically)."""

import pytest

from stow_recommender.domain import Bin, Item
from stow_recommender.train import run_training


def make_item(width=10, height=10, depth=10, weight_kg=1.0, hazardous=False, item_id="T1"):
    return Item(item_id, width, height, depth, weight_kg, hazardous)


def make_bin(bin_type="medium", level=3, fill=0.0, has_hazardous=False, bin_id="B1"):
    from stow_recommender.config import BIN_SIZES

    w, h, d = BIN_SIZES[bin_type]
    return Bin(bin_id, level, bin_type, w, h, d, fill * w * h * d, has_hazardous)


@pytest.fixture(scope="session")
def trained():
    """A smaller training run (fast), shared by all tests that need a model."""
    model, report, _ = run_training(n_train=6000, n_holdout=1000, n_scenarios=150)
    return model, report
