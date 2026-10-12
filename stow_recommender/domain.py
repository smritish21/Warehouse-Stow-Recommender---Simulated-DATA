"""The two things we reason about: items and bins."""

from dataclasses import dataclass

from stow_recommender.config import BIN_SIZES, POD_LAYOUT, TINY_MAX_DIM_CM


@dataclass(frozen=True)
class Item:
    item_id: str
    width: float   # cm
    height: float  # cm
    depth: float   # cm
    weight_kg: float
    hazardous: bool = False  # can break or spill

    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth

    @property
    def dims_sorted(self) -> tuple:
        # Items can be rotated, so we compare sorted sides, smallest first.
        return tuple(sorted((self.width, self.height, self.depth)))

    @property
    def is_tiny(self) -> bool:
        return max(self.width, self.height, self.depth) < TINY_MAX_DIM_CM


@dataclass(frozen=True)
class Bin:
    bin_id: str
    level: int
    bin_type: str  # "small", "medium" or "large"
    width: float
    height: float
    depth: float
    used_volume: float = 0.0      # space already taken by other items
    has_hazardous: bool = False   # already contains a hazardous item

    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth

    @property
    def free_volume(self) -> float:
        return self.volume - self.used_volume

    @property
    def fill_ratio(self) -> float:
        return self.used_volume / self.volume

    @property
    def dims_sorted(self) -> tuple:
        return tuple(sorted((self.width, self.height, self.depth)))


def build_pod(pod_id: str, fill_ratios=None, hazardous_bins=frozenset()) -> list[Bin]:
    """Create all bins of one pod from POD_LAYOUT.

    fill_ratios: optional {bin_id: fraction already used}
    hazardous_bins: bin_ids that already hold a hazardous item
    """
    fill_ratios = fill_ratios or {}
    bins = []
    for level, bin_types in POD_LAYOUT.items():
        for column, bin_type in enumerate(bin_types, start=1):
            bin_id = f"{pod_id}-L{level}-C{column}"
            w, h, d = BIN_SIZES[bin_type]
            bins.append(Bin(
                bin_id=bin_id,
                level=level,
                bin_type=bin_type,
                width=w, height=h, depth=d,
                used_volume=fill_ratios.get(bin_id, 0.0) * w * h * d,
                has_hazardous=bin_id in hazardous_bins,
            ))
    return bins
