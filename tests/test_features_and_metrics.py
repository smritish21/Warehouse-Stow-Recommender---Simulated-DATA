"""Features, the simulated truth, and ranking metrics, checked against hand calculations."""

import numpy as np
import pytest

from conftest import make_bin, make_item
from stow_recommender.evaluate import dcg, ndcg_at_k
from stow_recommender.features import FEATURE_NAMES, features_frame, placement_features
from stow_recommender.simulate import (
    FILL_TARGET,
    generate_history,
    hidden_factor,
    true_probability,
)


# ---------- Features ----------

def test_fill_after_adds_item_volume():
    item = make_item(10, 10, 21)                       # 2,100 cm³
    f = placement_features(item, make_bin("medium", fill=0.5))  # medium = 42,000 cm³
    assert f["fill_before"] == pytest.approx(0.5)
    assert f["fill_after"] == pytest.approx(0.55)


def test_clearance_uses_best_rotation():
    # medium bin sorted sides (30, 35, 40); item sorted (10, 20, 33) -> gaps 20, 15, 7
    assert placement_features(make_item(33, 10, 20), make_bin("medium"))["min_clearance_cm"] == 7


def test_features_frame_has_fixed_columns():
    frame = features_frame(make_item(), [make_bin(bin_id="A"), make_bin(bin_id="B")])
    assert list(frame.columns) == FEATURE_NAMES
    assert len(frame) == 2


# ---------- The simulated truth encodes the stow rules ----------

def test_comfortable_fill_beats_empty_and_packed():
    plenty_of_space = 10
    comfortable = true_probability(FILL_TARGET, plenty_of_space)
    assert comfortable > true_probability(0.10, plenty_of_space)  # big bin, tiny item: wasteful
    assert comfortable > true_probability(0.98, plenty_of_space)  # packed full


def test_exact_fit_is_bad_even_at_good_fill():
    assert true_probability(FILL_TARGET, 0.2) < true_probability(FILL_TARGET, 5)


def test_hidden_factor_is_repeatable_and_bounded():
    assert hidden_factor("I1", "B1") == hidden_factor("I1", "B1")
    values = [hidden_factor(f"I{i}", "B1") for i in range(500)]
    assert min(values) >= 0.75 and max(values) <= 1.25
    assert np.mean(values) == pytest.approx(1.0, abs=0.05)


def test_history_is_reproducible():
    a, b = generate_history(200, seed=5), generate_history(200, seed=5)
    assert a.equals(b)
    assert set(a["good_stow"].unique()) <= {0, 1}


# ---------- Ranking metrics ----------

def test_dcg_hand_calculation():
    # 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.2619 + 0.5
    assert dcg([3, 2, 1]) == pytest.approx(4.7619, abs=1e-4)


def test_ndcg_perfect_ranking_is_1():
    assert ndcg_at_k([0.9, 0.5, 0.1], scores=[3, 2, 1]) == pytest.approx(1.0)


def test_ndcg_reversed_ranking_is_lower():
    assert ndcg_at_k([0.9, 0.5, 0.1, 0.0], scores=[0, 1, 2, 3]) < 0.5


def test_ndcg_only_looks_at_top_k():
    # Swapping the last two of four bins doesn't change NDCG@2
    truth = [0.9, 0.8, 0.3, 0.1]
    assert ndcg_at_k(truth, [4, 3, 1, 2], k=2) == pytest.approx(1.0)
