"""Show a recommendation for a random item and pod in the terminal.

Run:  python -m stow_recommender.demo            (random example)
      python -m stow_recommender.demo --seed 7   (repeatable example)
"""

import argparse

import numpy as np

from stow_recommender.config import POD_LAYOUT
from stow_recommender.recommend import BEST_FIT, GOOD_FIT, load_score_fn, recommend
from stow_recommender.simulate import random_item, random_pod


def show(item, pod, recommendations) -> None:
    by_id = {r.bin_id: r for r in recommendations}
    best = [r for r in recommendations if r.status == BEST_FIT]
    best_rank = {r.bin_id: i + 1 for i, r in enumerate(best)}

    print(f"\nItem {item.item_id}: {item.width} x {item.height} x {item.depth} cm, "
          f"{item.weight_kg} kg, hazardous: {'yes' if item.hazardous else 'no'}"
          f"{', TINY' if item.is_tiny else ''}\n")
    print("Pod (top shelf first)    [1][2][3] = best fit   ok = good fit   XX = not allowed\n")
    bins_by_level = {}
    for b in pod:
        bins_by_level.setdefault(b.level, []).append(b)
    for level in sorted(POD_LAYOUT, reverse=True):
        cells = []
        for b in bins_by_level[level]:
            r = by_id[b.bin_id]
            mark = f"[{best_rank[b.bin_id]}]" if r.status == BEST_FIT else (" ok" if r.status == GOOD_FIT else " XX")
            haz = "*" if b.has_hazardous else " "
            cells.append(f"{mark} {b.bin_type[0].upper()} {b.fill_ratio:4.0%}{haz}")
        print(f"  Level {level} | " + " | ".join(cells))
    print("\n  S/M/L = bin size, % = how full now, * = holds a hazardous item\n")

    for i, r in enumerate(best, start=1):
        print(f"  Best fit {i}: {r.bin_id}  (score {r.score:.2f})")
    red = [r for r in recommendations if r.reasons]
    if red:
        reasons = {}
        for r in red:
            for reason in r.reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        print("\n  Not allowed:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {count:2d} bins: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    item, pod = random_item(rng, "DEMO"), random_pod(rng, "P1")
    show(item, pod, recommend(item, pod, load_score_fn()))


if __name__ == "__main__":
    main()
