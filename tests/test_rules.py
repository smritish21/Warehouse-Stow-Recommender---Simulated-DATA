"""Hard rules, one behaviour per test."""

import pytest

from conftest import make_bin, make_item
from stow_recommender.config import MAX_WEIGHT_KG_BY_LEVEL
from stow_recommender.rules import fits, is_allowed, violations


# ---------- Fit ----------

def test_small_item_in_empty_bin_is_allowed():
    assert is_allowed(make_item(), make_bin())


def test_item_longer_than_bin_does_not_fit():
    item = make_item(width=60)  # longest bin side is 50 cm
    assert "does not fit" in violations(item, make_bin("large", level=1))


def test_rotated_item_fits():
    # 38 cm tall doesn't fit a 30 cm high medium bin upright, but fits lying down (40 cm deep)
    item = make_item(width=20, height=38, depth=10)
    assert fits(item, make_bin("medium"))


def test_item_does_not_fit_when_bin_is_too_full():
    item = make_item(width=20, height=20, depth=20)          # 8,000 cm³
    almost_full = make_bin("medium", fill=0.9)                # 4,200 cm³ free
    assert not fits(item, almost_full)


# ---------- Weight by level ----------

@pytest.mark.parametrize("level", [5, 6])
def test_heavy_item_not_allowed_on_top_levels(level):
    reasons = violations(make_item(weight_kg=8), make_bin("small", level=level))
    assert any("too heavy" in r for r in reasons)


def test_heavy_item_allowed_at_bottom():
    assert is_allowed(make_item(weight_kg=20), make_bin("large", level=1))


@pytest.mark.parametrize("level, limit", MAX_WEIGHT_KG_BY_LEVEL.items())
def test_weight_exactly_at_limit_is_allowed(level, limit):
    # Boundary value: the limit itself is allowed, anything above is not.
    bin_type = "large" if level <= 2 else "medium" if level == 3 else "small"
    assert is_allowed(make_item(5, 5, 5, weight_kg=limit), make_bin(bin_type, level=level))
    assert not is_allowed(make_item(5, 5, 5, weight_kg=limit + 0.01), make_bin(bin_type, level=level))


# ---------- Hazardous ----------

def test_two_hazardous_items_never_share_a_bin():
    reasons = violations(make_item(hazardous=True), make_bin(has_hazardous=True, fill=0.2))
    assert "bin already holds a hazardous item" in reasons


def test_hazardous_item_allowed_in_bin_without_hazardous():
    assert is_allowed(make_item(hazardous=True), make_bin(fill=0.2))


def test_normal_item_allowed_next_to_hazardous():
    assert is_allowed(make_item(hazardous=False), make_bin(has_hazardous=True, fill=0.2))


# ---------- Tiny items ----------

@pytest.mark.parametrize("bin_type, level", [("medium", 3), ("large", 1)])
def test_tiny_item_not_allowed_in_bigger_bins(bin_type, level):
    tiny = make_item(2, 1, 1, weight_kg=0.01)
    assert "tiny item needs a small bin" in violations(tiny, make_bin(bin_type, level=level))


def test_tiny_item_allowed_in_small_bin():
    assert is_allowed(make_item(2, 1, 1, weight_kg=0.01), make_bin("small", level=5))


def test_item_of_exactly_3cm_is_not_tiny():
    # Boundary: tiny means longest side UNDER 3 cm
    assert is_allowed(make_item(3, 1, 1, weight_kg=0.01), make_bin("large", level=1))


# ---------- Several problems at once ----------

def test_all_broken_rules_are_reported():
    item = make_item(2, 1, 1, weight_kg=10, hazardous=True)  # tiny, heavy, hazardous
    reasons = violations(item, make_bin("medium", level=4, fill=0.2, has_hazardous=True))
    assert len(reasons) == 2  # hazardous + tiny (10 kg is within the 12 kg limit on level 4)
    reasons = violations(item, make_bin("small", level=6, fill=0.2, has_hazardous=True))
    assert len(reasons) == 2  # too heavy + hazardous (small bin is fine for tiny)
