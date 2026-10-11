"""Combine the model's ranking with the hard rules into the final recommendation.

1. Hard rules split the bins into allowed and not allowed ("red", with reasons).
2. The model scores ONLY the allowed bins.
3. The top BEST_FIT_COUNT allowed bins are "best fit", the rest "good fit".

The model never sees a red bin, so it can never recommend one.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib

from stow_recommender.config import BEST_FIT_COUNT, MODEL_PATH
from stow_recommender.features import features_frame
from stow_recommender.rules import violations

BEST_FIT, GOOD_FIT, NOT_ALLOWED = "best_fit", "good_fit", "not_allowed"


@dataclass
class Recommendation:
    bin_id: str
    level: int
    status: str
    score: float | None = None          # model score, only for allowed bins
    reasons: list[str] = field(default_factory=list)  # why a bin is not allowed


def recommend(item, bins, score_fn) -> list[Recommendation]:
    """Best fit first (highest score), then good fit, then not allowed."""
    allowed, red = [], []
    for b in bins:
        reasons = violations(item, b)
        if reasons:
            red.append(Recommendation(b.bin_id, b.level, NOT_ALLOWED, reasons=reasons))
        else:
            allowed.append(b)

    ranked = []
    if allowed:
        scores = score_fn(item, allowed)
        order = sorted(range(len(allowed)), key=lambda i: -scores[i])
        for rank, i in enumerate(order):
            status = BEST_FIT if rank < BEST_FIT_COUNT else GOOD_FIT
            ranked.append(Recommendation(allowed[i].bin_id, allowed[i].level, status, float(scores[i])))
    return ranked + red


def model_score_fn(model):
    """Wrap a trained classifier: score = predicted chance of a good stow."""
    def score(item, bins):
        return model.predict_proba(features_frame(item, bins))[:, 1]
    return score


def load_score_fn(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError("No trained model found. Run: python -m stow_recommender.train")
    return model_score_fn(joblib.load(path)["model"])
