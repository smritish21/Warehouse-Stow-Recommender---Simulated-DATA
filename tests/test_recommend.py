"""The final recommendation: model ranking + hard-rule safety layer."""

import numpy as np

from conftest import make_bin, make_item
from stow_recommender.config import BEST_FIT_COUNT
from stow_recommender.domain import build_pod
from stow_recommender.recommend import BEST_FIT, GOOD_FIT, NOT_ALLOWED, model_score_fn, recommend
from stow_recommender.rules import is_allowed
from stow_recommender.simulate import random_item, random_pod


def by_fill_score(item, bins):
    """A simple stand-in scorer for tests that don't need a trained model."""
    return [b.fill_ratio for b in bins]


def test_every_bin_gets_exactly_one_status():
    item, pod = make_item(), build_pod("P")
    recs = recommend(item, pod, by_fill_score)
    assert sorted(r.bin_id for r in recs) == sorted(b.bin_id for b in pod)


def test_top_three_are_best_fit_in_score_order():
    pod = build_pod("P", fill_ratios={"P-L3-C1": 0.5, "P-L3-C2": 0.4, "P-L3-C3": 0.3})
    recs = recommend(make_item(), pod, by_fill_score)
    best = [r for r in recs if r.status == BEST_FIT]
    assert [r.bin_id for r in best] == ["P-L3-C1", "P-L3-C2", "P-L3-C3"]
    assert len(best) == BEST_FIT_COUNT


def test_order_is_best_then_good_then_not_allowed():
    item = make_item(weight_kg=8)  # too heavy for levels 5-6
    statuses = [r.status for r in recommend(item, build_pod("P"), by_fill_score)]
    rank = {BEST_FIT: 0, GOOD_FIT: 1, NOT_ALLOWED: 2}
    assert statuses == sorted(statuses, key=rank.get)


def test_not_allowed_bins_have_reasons_and_no_score():
    item = make_item(weight_kg=8)
    for r in recommend(item, build_pod("P"), by_fill_score):
        if r.status == NOT_ALLOWED:
            assert r.reasons and r.score is None
        else:
            assert not r.reasons and r.score is not None


def test_no_allowed_bin_means_no_recommendation():
    huge = make_item(80, 80, 80, weight_kg=50)
    recs = recommend(huge, build_pod("P"), by_fill_score)
    assert all(r.status == NOT_ALLOWED for r in recs)


def test_scorer_only_sees_allowed_bins():
    seen = []

    def spy(item, bins):
        seen.extend(bins)
        return [0.5] * len(bins)

    item = make_item(weight_kg=8)
    recommend(item, build_pod("P"), spy)
    assert seen and all(is_allowed(item, b) for b in seen)


def test_model_can_never_recommend_a_forbidden_bin(trained):
    # Safety property, checked on 300 random items and pods with the REAL model:
    # every bin marked best or good fit must pass all hard rules.
    model, _ = trained
    score_fn = model_score_fn(model)
    rng = np.random.default_rng(123)
    for i in range(300):
        item, pod = random_item(rng, f"X{i}"), random_pod(rng, f"Q{i}")
        bins = {b.bin_id: b for b in pod}
        for r in recommend(item, pod, score_fn):
            if r.status != NOT_ALLOWED:
                assert is_allowed(item, bins[r.bin_id]), (item, r)


def test_model_prefers_comfortable_fit_over_exact_fit(trained):
    # Same item, three allowed bins: nearly empty, comfortably full, and packed tight.
    model, _ = trained
    item = make_item(20, 15, 14)  # 4,200 cm³ = 10% of a medium bin
    bins = [
        make_bin("medium", fill=0.05, bin_id="almost_empty"),   # ends at 15% full
        make_bin("medium", fill=0.55, bin_id="comfortable"),    # ends at 65% full
        make_bin("medium", fill=0.88, bin_id="packed"),         # ends at 98% full
    ]
    best = recommend(item, bins, model_score_fn(model))[0]
    assert best.bin_id == "comfortable"
