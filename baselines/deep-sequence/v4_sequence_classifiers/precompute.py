"""Build the per-split feature cache for v4_sequence_classifiers.

For each split in {train, val, test} and a chosen feature set, emits:

    logs/anticipatory_classifier/embeddings_cache/{split}_seq_{feature_set}.pt

Each file is a dict:
    {
      "dyn":           torch.Tensor [N, T=20, D_dyn]  float32  — raw, unnormalised
      "static":        torch.Tensor [N, D_static]      float32  — raw, unnormalised
      "y":             torch.Tensor [N]                long
      "meta":          list[dict] of scene_name / ped_id / frame_current
      "dyn_names":     list[str]          length D_dyn
      "dyn_groups":    list[str]          length D_dyn
      "static_names":  list[str]          length D_static
      "feature_set":   "seq_core" | "seq_core_ctx"
      "signal_missing_scenes": list[str]
      "windows_total": int
      "skipped_no_hist": int
    }

Normalisation is intentionally left to training time so the stats are always
learned on the *train* split of the current run (cleaner audit trail).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_pipeline.split_utils import load_split  # noqa: E402
from models.anticipatory_classifier.common.data_utils import filter_gt_to_split  # noqa: E402
from models.anticipatory_classifier.common.progress import pbar  # noqa: E402
from models.anticipatory_classifier.versions.v4_sequence_classifiers import features as f4  # noqa: E402


FEATURE_SETS = ("seq_core", "seq_core_ctx")


def _process_split(
    split_name: str,
    gt_split: pd.DataFrame,
    traj_dir: Path,
    builder: f4.SeqFeatureBuilder,
    fps: float,
    disable_progress: bool = False,
) -> Dict:
    dyn_rows: List[np.ndarray] = []
    static_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    meta_rows: List[Dict] = []
    missing_scenes: List[str] = []

    scene_groups = list(gt_split.groupby("scene_name", sort=False))
    it = pbar(scene_groups, desc=f"[{split_name}] scenes", unit="scene",
              disable=disable_progress)
    for scene_name, g in it:
        ctx = f4.load_scene_context(str(scene_name), traj_dir, fps=fps)
        if ctx.signal_missing:
            missing_scenes.append(str(scene_name))
        builder.set_scene(ctx)

        # Per-row progress is noisy on Spartan; keep it scene-level. Inner
        # loop uses enumerate so we can emit a single "N rows" line at end.
        for _, row in g.iterrows():
            try:
                dyn, static = builder.compute_row(row)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] feature compute failed on "
                      f"({scene_name}, ped={row['ped_id']}, "
                      f"f={row['frame_current']}): {e}")
                T, D_dyn, D_static = f4.feature_shapes(builder.include_context)
                dyn = np.full((T, D_dyn), np.nan, dtype=np.float32)
                static = np.full((D_static,), np.nan, dtype=np.float32)
            dyn_rows.append(dyn)
            static_rows.append(static)
            y_rows.append(int(row["label"]))
            meta_rows.append({
                "scene_name": str(row["scene_name"]),
                "ped_id": int(row["ped_id"]),
                "frame_current": int(row["frame_current"]),
            })

    T, D_dyn, D_static = f4.feature_shapes(builder.include_context)
    if dyn_rows:
        Xdyn = np.stack(dyn_rows).astype(np.float32)
        Xst = np.stack(static_rows).astype(np.float32)
    else:
        Xdyn = np.zeros((0, T, D_dyn), dtype=np.float32)
        Xst = np.zeros((0, D_static), dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)

    return {
        "dyn": torch.from_numpy(Xdyn),
        "static": torch.from_numpy(Xst),
        "y": torch.from_numpy(y),
        "meta": meta_rows,
        "dyn_names": list(builder.dyn_names),
        "dyn_groups": list(builder.dyn_groups),
        "static_names": list(builder.static_names),
        "feature_set": "seq_core_ctx" if builder.include_context else "seq_core",
        "signal_missing_scenes": missing_scenes,
        "windows_total": len(y_rows),
        "skipped_no_hist": int(builder.counts.failed),
    }


def main() -> int:
    repo = _REPO
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=repo / "dataset/FI")
    p.add_argument("--gt-path", type=Path,
                   default=repo / "dataset/FI/processed_data/jaywalk_ground_truth.csv")
    p.add_argument("--zones-csv", type=Path,
                   default=repo / "dataset/FI/derived_data/map/FI_int_all_zones.csv")
    p.add_argument("--traj-dir", type=Path,
                   default=repo / "dataset/FI/derived_data/traj")
    p.add_argument("--cache-dir", type=Path,
                   default=repo / "logs/anticipatory_classifier/embeddings_cache")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--feature-set", choices=FEATURE_SETS, default="seq_core")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--force", action="store_true",
                   help="Re-compute even if a cache file already exists.")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()

    for pth in (args.gt_path, args.zones_csv, args.traj_dir):
        if not pth.exists():
            print(f"ERROR: missing {pth}", file=sys.stderr)
            return 1

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GT    : {args.gt_path}")
    gt_all = pd.read_csv(args.gt_path)
    print(f"  {len(gt_all)} rows")

    print(f"Loading split : {args.dataset_root}")
    splits = load_split(args.dataset_root)
    for k, v in splits.items():
        print(f"  {k}: {len(v)} scenes")

    print(f"Loading zones : {args.zones_csv}")
    zones = f4.load_zones(args.zones_csv)
    print(f"  walking_areas={len(zones.walking_areas)} "
          f"crosswalks={list(zones.crosswalks)} "
          f"no_ped_areas={len(zones.no_ped_areas)} "
          f"waiting_zones={len(zones.waiting_zones)}")

    include_context = (args.feature_set == "seq_core_ctx")
    print(f"Feature set   : {args.feature_set}  (include_context={include_context})")

    for split_name in args.splits:
        out_path = args.cache_dir / f"{split_name}_seq_{args.feature_set}.pt"
        if out_path.exists() and not args.force:
            print(f"\n[{split_name}] cache exists at {out_path} — skip (use --force to rebuild)")
            continue

        gt_split = filter_gt_to_split(gt_all, splits[split_name])
        print(f"\n--- {split_name}: {len(gt_split)} rows ---")

        builder = f4.SeqFeatureBuilder(zones, fps=args.fps,
                                       include_context=include_context)
        cache = _process_split(
            split_name, gt_split, args.traj_dir, builder,
            fps=args.fps, disable_progress=args.no_progress,
        )
        torch.save(cache, str(out_path))
        n = int(cache["y"].shape[0])
        pos = int(cache["y"].sum())
        pos_rate = pos / max(n, 1)
        print(f"  saved {out_path}  "
              f"dyn={tuple(cache['dyn'].shape)}  "
              f"static={tuple(cache['static'].shape)}  "
              f"pos_rate={pos_rate:.3f}  "
              f"missing_signal_scenes={cache['signal_missing_scenes']}")

    print("\nprecompute-seq done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
