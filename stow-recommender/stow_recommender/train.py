"""Train the stow model and compare it with simple rules of thumb.

Run:  python -m stow_recommender.train

Steps:
1. Simulate 20,000 past stows (item, chosen bin, good/bad outcome) for training.
2. Train a gradient-boosting classifier to predict "good stow" from bin/item features.
3. On 500 NEW scenarios, rank all allowed bins and compare the ranking quality
   with three baselines: tightest fit, most free space, and random.
4. Save the model, metrics and a plot of what the model learned.
"""

import json

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from stow_recommender.config import MODEL_PATH, RANDOM_STATE, REPORTS_DIR
from stow_recommender.evaluate import evaluate_ranker
from stow_recommender.features import FEATURE_NAMES, features_frame
from stow_recommender.recommend import model_score_fn
from stow_recommender.simulate import generate_history, generate_scenarios, true_probability

N_TRAIN_EVENTS = 20_000
N_HOLDOUT_EVENTS = 5_000
N_EVAL_SCENARIOS = 500


def train_model(history):
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=RANDOM_STATE
    )
    model.fit(history[FEATURE_NAMES], history["good_stow"])
    return model


# ---------- Baselines: simple rules of thumb the model must beat ----------

def tightest_fit_scores(item, bins):
    """Classic bin packing: fill bins as full as possible."""
    return features_frame(item, bins)["fill_after"].to_numpy()


def most_space_scores(item, bins):
    """The opposite: always pick the emptiest bin."""
    return -features_frame(item, bins)["fill_after"].to_numpy()


def random_scores_factory(seed):
    rng = np.random.default_rng(seed)
    return lambda item, bins: rng.random(len(bins))


def run_training(n_train=N_TRAIN_EVENTS, n_holdout=N_HOLDOUT_EVENTS,
                 n_scenarios=N_EVAL_SCENARIOS, seed=RANDOM_STATE):
    """Different seeds for training, holdout and evaluation = no overlap between them."""
    
    history = generate_history(n_train, seed)
    holdout = generate_history(n_holdout, seed + 1)
    scenarios = generate_scenarios(n_scenarios, seed + 2)

    model = train_model(history)
    holdout_proba = model.predict_proba(holdout[FEATURE_NAMES])[:, 1]

    rankers = {
        "model": model_score_fn(model),
        "tightest_fit": tightest_fit_scores,
        "most_space": most_space_scores,
        "random": random_scores_factory(seed + 3),
    }
    report = {
        "n_train_events": len(history),
        "good_stow_rate_train": float(history["good_stow"].mean()),
        "holdout_roc_auc": float(roc_auc_score(holdout["good_stow"], holdout_proba)),
        "ranking": {name: evaluate_ranker(scenarios, fn) for name, fn in rankers.items()},
    }
    return model, report, holdout.assign(model_proba=holdout_proba)


def save_fit_curve_plot(holdout) -> str:
    """Average good-stow rate by how full the bin ends up: data vs model vs truth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = holdout.copy()
    df["truth"] = [true_probability(f, c) for f, c in zip(df["fill_after"], df["min_clearance_cm"])]
    df["fill_bucket"] = (df["fill_after"] // 0.05) * 0.05 + 0.025
    curve = df.groupby("fill_bucket")[["good_stow", "model_proba", "truth"]].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve.index, curve["good_stow"], "o", alpha=0.5, label="observed outcomes (noisy)")
    ax.plot(curve.index, curve["model_proba"], linewidth=2, label="model prediction")
    ax.plot(curve.index, curve["truth"], "--", color="grey", label="hidden truth (average)")
    ax.set_xlabel("How full the bin is after stowing")
    ax.set_ylabel("Chance of a good stow")
    ax.set_title("What the model learned: comfortable fit beats empty or packed bins")
    ax.legend()
    fig.tight_layout()
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "learned_fit_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def main() -> None:
    model, report, holdout = run_training()

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURE_NAMES, "sklearn_version": sklearn.__version__},
        MODEL_PATH,
    )
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    save_fit_curve_plot(holdout)

    print(f"\nTrained on {report['n_train_events']} simulated stows "
          f"({report['good_stow_rate_train']:.0%} went well)")
    print(f"Holdout ROC AUC: {report['holdout_roc_auc']:.3f}")
    print("\nRanking quality on 500 new scenarios (higher is better):")
    print(f"  {'method':14s} {'NDCG@3':>8s} {'top pick in true top 3':>24s}")
    for name, m in report["ranking"].items():
        print(f"  {name:14s} {m['ndcg@3']:8.3f} {m['top1_in_true_top3']:24.0%}")
    print(f"\nSaved model to {MODEL_PATH} and reports to {REPORTS_DIR}/")
    print("Try it: python -m stow_recommender.demo")


if __name__ == "__main__":
    main()
