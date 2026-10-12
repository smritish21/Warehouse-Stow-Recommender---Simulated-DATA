"""Quality gates: the model must clearly beat simple rules of thumb."""


def test_model_beats_every_baseline(trained):
    _, report = trained
    ranking = report["ranking"]
    for baseline in ["tightest_fit", "most_space", "random"]:
        assert ranking["model"]["ndcg@3"] > ranking[baseline]["ndcg@3"] + 0.2, baseline


def test_model_ranking_is_good_in_absolute_terms(trained):
    _, report = trained
    assert report["ranking"]["model"]["ndcg@3"] >= 0.85
    assert report["ranking"]["model"]["top1_in_true_top3"] >= 0.6


def test_model_separates_good_and_bad_stows(trained):
    _, report = trained
    assert report["holdout_roc_auc"] >= 0.8


def test_tightest_fit_is_not_a_good_strategy(trained):
    # Documents the key insight: classic "pack it tight" bin packing is
    # no better than random here, because tightly packed bins are hard to pick.
    _, report = trained
    ranking = report["ranking"]
    assert ranking["tightest_fit"]["ndcg@3"] < ranking["random"]["ndcg@3"] + 0.05
