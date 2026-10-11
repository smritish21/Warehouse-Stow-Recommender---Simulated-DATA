"""Ranking metrics: how good is the ORDER in which bins are recommended?"""

import numpy as np

from stow_recommender.rules import is_allowed
from stow_recommender.simulate import true_good_stow_probability


def dcg(relevances) -> float:
    """Discounted cumulative gain: good bins count more when ranked higher."""
    relevances = np.asarray(relevances, dtype=float)
    positions = np.arange(2, len(relevances) + 2)
    return float(np.sum(relevances / np.log2(positions)))


def ndcg_at_k(true_relevance, scores, k: int = 3) -> float:
    """1.0 = the top k are exactly the best possible bins, in the best order."""
    true_relevance = np.asarray(true_relevance, dtype=float)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ideal = np.sort(true_relevance)[::-1]
    best = dcg(ideal[:k])
    return dcg(true_relevance[order][:k]) / best if best > 0 else 0.0


def evaluate_ranker(scenarios, score_fn, k: int = 3) -> dict:
    """Score a ranking method on scenarios where we know the true quality of every bin.

    score_fn(item, allowed_bins) -> one score per bin (higher = better)
    """
    ndcgs, top1_in_true_topk = [], []
    for item, pod in scenarios:
        allowed = [b for b in pod if is_allowed(item, b)]
        truth = np.array([true_good_stow_probability(item, b) for b in allowed])
        scores = np.asarray(score_fn(item, allowed), dtype=float)
        ndcgs.append(ndcg_at_k(truth, scores, k))
        top1 = int(np.argmax(scores))
        true_topk = set(np.argsort(-truth, kind="stable")[:k])
        top1_in_true_topk.append(top1 in true_topk)
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"top1_in_true_top{k}": float(np.mean(top1_in_true_topk)),
    }
