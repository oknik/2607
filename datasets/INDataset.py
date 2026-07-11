from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairedSample:
    sample_id: str
    label: int
    c_path: Path
    g_path: Path


def build_samples(data_dir: str | Path) -> list[PairedSample]:
    """Parse id-label-mode.png files and pair C/G modalities by id."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_path}")

    grouped: dict[str, dict[str, object]] = {}
    for path in sorted(data_path.glob("*.png")):
        parts = path.stem.split("-")
        if len(parts) != 3:
            continue
        sample_id, label_text, mode = parts
        if mode not in {"C", "G"}:
            continue
        item = grouped.setdefault(sample_id, {"labels": set(), "paths": {}})
        item["labels"].add(int(label_text))
        item["paths"][mode] = path

    samples: list[PairedSample] = []
    bad: list[str] = []
    for sample_id, item in grouped.items():
        labels = item["labels"]
        paths = item["paths"]
        if len(labels) != 1 or set(paths) != {"C", "G"}:
            bad.append(sample_id)
            continue
        samples.append(
            PairedSample(
                sample_id=sample_id,
                label=next(iter(labels)),
                c_path=paths["C"],
                g_path=paths["G"],
            )
        )

    if bad:
        raise ValueError(f"Found {len(bad)} invalid C/G pairs, examples: {bad[:5]}")
    if not samples:
        raise ValueError(f"No paired samples found in {data_path}")
    return samples


class INDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[PairedSample],
        transform: Callable | None = None,
        label_map: dict[int, int] | None = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.label_map = label_map or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        c_img = Image.open(sample.c_path).convert("RGB")
        g_img = Image.open(sample.g_path).convert("RGB")
        if self.transform is not None:
            c_img = self.transform(c_img)
            g_img = self.transform(g_img)
        label = self.label_map.get(sample.label, sample.label)
        return {
            "c": c_img,
            "g": g_img,
            "label": label,
            "sample_id": sample.sample_id,
        }
