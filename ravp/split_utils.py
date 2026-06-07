"""
Helpers for loading the canonical scene-level train/val/test split of a
FLUID intersection dataset.

The canonical split lives at
``<dataset_root>/processed_data/split_manifest.json`` and is produced by
``data_pipeline/determine_splits.py``. Every script that needs to know which
scene belongs to which split should read it through the helpers here so the
partition stays consistent across data formatting, model training, and
evaluation.
"""
import json
from pathlib import Path
from typing import Dict, List

SPLIT_NAMES = ("train", "val", "test")


def manifest_path(dataset_root):
    # type: (Path) -> Path
    return Path(dataset_root) / "processed_data" / "split_manifest.json"


def load_split(dataset_root):
    # type: (Path) -> Dict[str, List[str]]
    """Return ``{"train": [...], "val": [...], "test": [...]}`` for a dataset.

    Parameters
    ----------
    dataset_root
        Path to ``dataset/<DATASET>`` (e.g., ``dataset/FI``).

    Raises
    ------
    FileNotFoundError
        If the manifest has not been created yet.
    ValueError
        If the manifest is missing a required split or contains duplicate scenes.
    """
    path = manifest_path(dataset_root)
    if not path.exists():
        raise FileNotFoundError(
            "split_manifest.json not found at {p}. "
            "Run data_pipeline/determine_splits.py --dataset <NAME> --write {p} first.".format(p=path)
        )
    with open(str(path), "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    splits = {name: list(payload.get(name, [])) for name in SPLIT_NAMES}
    for name in SPLIT_NAMES:
        if not isinstance(splits[name], list):
            raise ValueError("Manifest {p}: '{n}' must be a list of scene names.".format(p=path, n=name))

    seen = set()
    for name in SPLIT_NAMES:
        for scene in splits[name]:
            if scene in seen:
                raise ValueError(
                    "Manifest {p}: scene '{s}' appears in more than one split.".format(p=path, s=scene)
                )
            seen.add(scene)

    return splits


def scene_to_split(scene_name, dataset_root):
    # type: (str, Path) -> str
    """Return which split (``train`` / ``val`` / ``test``) a scene belongs to.

    Returns an empty string for scenes that aren't listed in the manifest
    (caller decides whether that is an error or just "skip this scene").
    """
    for split, scenes in load_split(dataset_root).items():
        if scene_name in scenes:
            return split
    return ""
