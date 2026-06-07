"""End-to-end training of one v4 sequence classifier.

Invoked via either
    python -m models.anticipatory_classifier.cli train-seq \
      --model {lstm,gru,tcn} --feature-set {seq_core,seq_core_ctx} --tag ...
or directly:
    python -m models.anticipatory_classifier.versions.v4_sequence_classifiers.train \
      --model lstm --feature-set seq_core --tag smoke --epochs 3 --limit-train 512

For each run this script:
  1. Loads the cached feature tensors (emitted by precompute.py)
  2. Fits z-score stats on train; normalises all three splits
  3. Trains the chosen encoder with weighted BCE + AdamW + early stopping on
     val Macro-F1 (threshold is tuned on val each epoch)
  4. Loads the best checkpoint, produces test metrics at the tuned threshold
  5. Writes checkpoint, metrics.json, tidy CSV, predictions_test.csv,
     config_resolved.json, train_log.jsonl, and TensorBoard events
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.anticipatory_classifier.common import metrics as m_utils  # noqa: E402
from models.anticipatory_classifier.common.progress import pbar  # noqa: E402
from models.anticipatory_classifier.versions.v4_sequence_classifiers import (  # noqa: E402
    dataset as v4data,
    models as v4models,
)

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    _TB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# RunPaths (kept for optional CLI integration)
# ---------------------------------------------------------------------------


@dataclass
class RunPaths:
    run_dir: Path
    ckpt_path: Path
    metrics_path: Path
    log_path: Path


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


DEFAULT_CONFIG: Dict = {
    "version": "v4_sequence_classifiers",
    "embeddings_cache_dir": "logs/anticipatory_classifier/embeddings_cache",
    "log_dir": "logs/anticipatory_classifier/v4_sequence_classifiers",
    "T": 20,
    "threshold_grid": [0.05, 0.95, 0.01],
    "training": {
        "epochs": 120,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 15,
        "seed": 42,
        "num_workers": 2,
        "log_step_every": 100,
    },
    "lstm": {"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "pool": "last"},
    "gru":  {"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "pool": "last"},
    "tcn":  {"channels": [64, 64, 128, 128], "kernel_size": 3,
             "dilations": [1, 2, 4, 8], "dropout": 0.1, "pool": "mean"},
}


def _load_config(path: Optional[Path]) -> Dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if path is not None and Path(path).exists():
        with open(path) as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO),
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def _run_epoch(model: nn.Module,
               loader: DataLoader,
               device: torch.device,
               loss_fn: nn.Module,
               optimiser: Optional[torch.optim.Optimizer] = None,
               writer=None,
               global_step: int = 0,
               log_step_every: int = 100,
               disable_progress: bool = False,
               desc: str = "") -> Tuple[float, np.ndarray, np.ndarray, int]:
    """Run one pass. If `optimiser` is None, eval mode (no grad)."""
    training = optimiser is not None
    model.train(training)

    total_loss, total_n = 0.0, 0
    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    it = pbar(loader, desc=desc, unit="batch", disable=disable_progress)
    for batch in it:
        dyn = batch["dyn"].to(device, non_blocking=True)
        stat = batch["static"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(dyn, stat, mask)
            loss = loss_fn(logits, y)

        if training:
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            global_step += 1
            if writer is not None and (global_step % log_step_every == 0):
                writer.add_scalar("loss/train_step", float(loss.item()), global_step)

        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        total_n += bs
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    probs = np.concatenate(probs_all) if probs_all else np.zeros(0, dtype=np.float32)
    # Cast y to int64 so sklearn.metrics.classification_report uses integer
    # class keys ("0", "1") — otherwise float y arrays yield "0.0"/"1.0" keys
    # and the jaywalk-class stats silently return zeros.
    ys = (np.concatenate(y_all) if y_all else np.zeros(0, dtype=np.float32)).astype(np.int64)
    avg_loss = total_loss / max(total_n, 1)
    return avg_loss, probs, ys, global_step


# ---------------------------------------------------------------------------
# Main train entry
# ---------------------------------------------------------------------------


def train(config: Dict,
          model_name: str,
          feature_set: str,
          run_paths: RunPaths,
          device: str = "auto",
          limit_train: Optional[int] = None,
          disable_progress: bool = False,
          disable_tensorboard: bool = False) -> Dict:
    t_start = time.time()
    dev = _resolve_device(device)
    seed = int(config["training"].get("seed", 42))
    _set_seed(seed)

    cache_dir = Path(config["embeddings_cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = _REPO / cache_dir
    log_dir = Path(config["log_dir"])
    if not log_dir.is_absolute():
        log_dir = _REPO / log_dir

    run_dir = Path(run_paths.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    print(f"[train-seq] model={model_name} feature_set={feature_set} device={dev}")
    print(f"[train-seq] run_dir={run_dir}")

    # --- load caches + normalise --------------------------------------------
    datasets, stats, meta = v4data.build_datasets(
        cache_dir, feature_set, splits=("train", "val", "test"),
    )
    ds_tr, ds_va, ds_te = datasets["train"], datasets["val"], datasets["test"]

    if limit_train is not None and limit_train > 0 and limit_train < len(ds_tr):
        idx = torch.randperm(len(ds_tr))[:limit_train]
        ds_tr = v4data.SequenceWindowDataset(
            dyn=ds_tr.dyn[idx], static=ds_tr.static[idx],
            y=ds_tr.y[idx], mask=ds_tr.mask[idx],
        )
        print(f"[train-seq] limit_train: using {len(ds_tr)}/"
              f"{len(datasets['train'])} training rows")

    y_tr = ds_tr.y.numpy()
    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    pos_weight = float(n_neg / max(n_pos, 1))
    print(f"[train-seq] n_train={len(ds_tr)} (pos={n_pos} neg={n_neg} "
          f"pos_weight={pos_weight:.3f})  n_val={len(ds_va)}  n_test={len(ds_te)}")

    # --- build model ---------------------------------------------------------
    d_in = ds_tr.dyn.shape[-1]
    d_static = ds_tr.static.shape[-1]
    T = ds_tr.dyn.shape[1]
    print(f"[train-seq] tensor shapes: dyn=(B,{T},{d_in})  static=(B,{d_static})")

    model = v4models.build_model(model_name, d_in=d_in, d_static=d_static,
                                 config=config).to(dev)
    n_params = v4models.count_parameters(model)
    print(f"[train-seq] model={model_name}  trainable_params={n_params:,}")

    # --- optimiser + loss ----------------------------------------------------
    tr = config["training"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr["lr"]),
        weight_decay=float(tr["weight_decay"]),
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))

    # --- dataloaders ---------------------------------------------------------
    bs = int(tr["batch_size"])
    nw = int(tr["num_workers"])
    pin = (dev.type == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True,
                       num_workers=nw, pin_memory=pin, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False,
                       num_workers=nw, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False,
                       num_workers=nw, pin_memory=pin)

    # --- tensorboard ---------------------------------------------------------
    tb_writer = None
    if _TB_AVAILABLE and not disable_tensorboard:
        tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        tb_writer.add_text("config", json.dumps(config, indent=2)[:10000])
        tb_writer.add_text("meta", json.dumps({
            "model": model_name, "feature_set": feature_set,
            "d_in": d_in, "d_static": d_static, "T": T,
            "n_params": n_params, "pos_weight": pos_weight,
            "device": str(dev),
        }, indent=2))

    # --- training loop -------------------------------------------------------
    thresh_grid = tuple(config.get("threshold_grid", [0.05, 0.95, 0.01]))
    epochs = int(tr["epochs"])
    patience = int(tr["patience"])
    log_step_every = int(tr.get("log_step_every", 100))
    log_path = run_dir / "train_log.jsonl"
    log_fh = open(log_path, "w")

    best_val_f1 = -1.0
    best_threshold = 0.5
    best_epoch = -1
    patience_ct = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, _, _, global_step = _run_epoch(
            model, dl_tr, dev, loss_fn, optimiser=opt,
            writer=tb_writer, global_step=global_step,
            log_step_every=log_step_every,
            disable_progress=disable_progress,
            desc=f"ep{epoch:03d} train",
        )
        val_loss, val_probs, val_y, _ = _run_epoch(
            model, dl_va, dev, loss_fn, optimiser=None,
            disable_progress=disable_progress,
            desc=f"ep{epoch:03d} val",
        )
        thr, val_macro = m_utils.sweep_threshold(val_probs, val_y, grid=thresh_grid)
        val_metrics = m_utils.compute_split_metrics(
            split="val", probs=val_probs, y_true=val_y, threshold=thr,
            windows_total=meta["windows_total"]["val"],
            skipped_no_hist=meta["skipped_no_hist"]["val"],
        )
        epoch_sec = time.time() - t0
        cur_lr = opt.param_groups[0]["lr"]

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_metrics.macro_f1,
            "val_pr_auc": val_metrics.pr_auc,
            "val_f1_jaywalk": val_metrics.f1_jaywalk,
            "val_precision_jaywalk": val_metrics.precision_jaywalk,
            "val_recall_jaywalk": val_metrics.recall_jaywalk,
            "val_threshold": thr,
            "lr": cur_lr,
            "epoch_sec": round(epoch_sec, 3),
            "samples_per_sec": round(len(ds_tr) / max(epoch_sec, 1e-3), 1),
        }
        log_fh.write(json.dumps(log_row) + "\n")
        log_fh.flush()

        if tb_writer is not None:
            tb_writer.add_scalar("loss/train", train_loss, epoch)
            tb_writer.add_scalar("loss/val", val_loss, epoch)
            tb_writer.add_scalar("val/macro_f1", val_metrics.macro_f1, epoch)
            tb_writer.add_scalar("val/pr_auc", val_metrics.pr_auc, epoch)
            tb_writer.add_scalar("val/f1_jaywalk", val_metrics.f1_jaywalk, epoch)
            tb_writer.add_scalar("val/precision_jaywalk", val_metrics.precision_jaywalk, epoch)
            tb_writer.add_scalar("val/recall_jaywalk", val_metrics.recall_jaywalk, epoch)
            tb_writer.add_scalar("val/best_threshold", thr, epoch)
            tb_writer.add_scalar("train/lr", cur_lr, epoch)
            tb_writer.add_scalar("misc/epoch_seconds", epoch_sec, epoch)
            tb_writer.add_scalar("misc/samples_per_second", log_row["samples_per_sec"], epoch)

        print(f"ep{epoch:03d}  loss_tr={train_loss:.4f} loss_va={val_loss:.4f}  "
              f"val_macro_f1={val_metrics.macro_f1:.4f}  "
              f"val_pr_auc={val_metrics.pr_auc:.4f}  "
              f"thr={thr:.2f}  {epoch_sec:.1f}s")

        improved = val_macro > best_val_f1 + 1e-5
        if improved:
            best_val_f1 = float(val_macro)
            best_threshold = float(thr)
            best_epoch = epoch
            patience_ct = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_name": model_name,
                "feature_set": feature_set,
                "config": config,
                "stats": stats.to_dict(),
                "threshold": best_threshold,
                "val_macro_f1": best_val_f1,
                "epoch": epoch,
                "d_in": d_in,
                "d_static": d_static,
                "T": T,
                "git_sha": _git_sha(),
            }, str(run_paths.ckpt_path))
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"[train-seq] early stopping at epoch {epoch} "
                      f"(best_val_macro_f1={best_val_f1:.4f} @ ep{best_epoch})")
                break

    log_fh.close()

    # --- load best + final eval ---------------------------------------------
    ckpt = torch.load(str(run_paths.ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(dev)

    def _eval_split(dl, split_name, total, skipped, thr):
        _, probs, y, _ = _run_epoch(
            model, dl, dev, loss_fn, optimiser=None,
            disable_progress=disable_progress,
            desc=f"final {split_name}",
        )
        sm = m_utils.compute_split_metrics(
            split=split_name, probs=probs, y_true=y, threshold=thr,
            windows_total=total, skipped_no_hist=skipped,
        )
        return sm, probs, y

    tuned = best_threshold
    per_split_metrics = []
    per_split_probs = {}
    for name, dl in (("train", dl_tr), ("val", dl_va), ("test", dl_te)):
        sm, probs, y = _eval_split(
            dl, name, meta["windows_total"][name], meta["skipped_no_hist"][name],
            tuned,
        )
        if name != "train":
            m_utils.print_split_report(sm)
        per_split_metrics.append(sm.to_dict())
        per_split_probs[name] = (probs, y)

    # 0.5-threshold comparison (test only)
    probs_te, y_te = per_split_probs["test"]
    sm_0p5 = m_utils.compute_split_metrics(
        split="test", probs=probs_te, y_true=y_te, threshold=0.5,
        windows_total=meta["windows_total"]["test"],
        skipped_no_hist=meta["skipped_no_hist"]["test"],
    )

    # --- artefacts -----------------------------------------------------------
    summary = {
        "version": "v4_sequence_classifiers",
        "model": model_name,
        "feature_set": feature_set,
        "tag": Path(run_paths.run_dir).name,
        "git_sha": _git_sha(),
        "device": str(dev),
        "n_params": n_params,
        "d_in": d_in,
        "d_static": d_static,
        "T": T,
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "tuned_threshold": tuned,
        "threshold_0.5_comparison_test": sm_0p5.to_dict(),
        "splits": per_split_metrics,
        "config": config,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with open(run_paths.metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train-seq] metrics: {run_paths.metrics_path}")

    with open(run_dir / "config_resolved.json", "w") as f:
        json.dump(config, f, indent=2)

    # tidy CSV
    tidy_path = run_dir / "metrics_tidy.csv"
    fields = ["model", "feature_set", "split", "macro_f1",
              "precision_jaywalk", "recall_jaywalk", "f1_jaywalk",
              "pr_auc", "threshold", "windows_valid", "jaywalk_count"]
    with open(tidy_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_split_metrics:
            w.writerow({
                "model": model_name, "feature_set": feature_set,
                **{k: r.get(k) for k in fields if k not in ("model", "feature_set")},
            })

    # test predictions
    pred_path = run_dir / "predictions_test.csv"
    probs_te, y_te = per_split_probs["test"]
    meta_te = meta["meta"].get("test", [])
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_name", "frame_current", "ped_id",
                    "y_true", "prob", "pred_tuned", "pred_0p5"])
        for i in range(len(probs_te)):
            mt = meta_te[i] if i < len(meta_te) else {}
            w.writerow([
                mt.get("scene_name", ""),
                mt.get("frame_current", ""),
                mt.get("ped_id", ""),
                int(y_te[i]),
                float(probs_te[i]),
                int(probs_te[i] >= tuned),
                int(probs_te[i] >= 0.5),
            ])

    if tb_writer is not None:
        tb_writer.add_text(
            "final",
            "```\n" + json.dumps({
                "model": model_name,
                "feature_set": feature_set,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1,
                "tuned_threshold": tuned,
                "test": per_split_metrics[-1],
                "test_at_0p5": sm_0p5.to_dict(),
            }, indent=2) + "\n```",
        )
        tb_writer.flush()
        tb_writer.close()

    print(f"[train-seq] done in {summary['elapsed_seconds']:.1f}s  "
          f"best_val_macro_f1={best_val_f1:.4f}  test_macro_f1="
          f"{per_split_metrics[-1]['macro_f1']:.4f}")
    return summary


# ---------------------------------------------------------------------------
# Standalone CLI (also used by cli.cmd_train_seq)
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=v4models.MODEL_NAMES, required=True)
    p.add_argument("--feature-set", choices=("seq_core", "seq_core_ctx"),
                   default="seq_core")
    p.add_argument("--config", type=Path, default=None,
                   help="Optional JSON overriding default hyperparams.")
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-tensorboard", action="store_true")
    args = p.parse_args()

    config = _load_config(args.config)
    if args.cache_dir is not None:
        config["embeddings_cache_dir"] = str(args.cache_dir)
    if args.log_dir is not None:
        config["log_dir"] = str(args.log_dir)
    for k_cli, k_cfg in [("epochs", "epochs"), ("batch_size", "batch_size"),
                         ("lr", "lr"), ("weight_decay", "weight_decay"),
                         ("patience", "patience"), ("seed", "seed"),
                         ("num_workers", "num_workers")]:
        v = getattr(args, k_cli.replace("-", "_"))
        if v is not None:
            config["training"][k_cfg] = v

    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_id = f"{ts}_{args.model}_{args.feature_set}"
    if args.tag:
        run_id += f"_{args.tag}"
    log_root = Path(config["log_dir"])
    if not log_root.is_absolute():
        log_root = _REPO / log_root
    run_dir = log_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_paths = RunPaths(
        run_dir=run_dir,
        ckpt_path=run_dir / "checkpoints" / "best.pt",
        metrics_path=run_dir / "metrics.json",
        log_path=run_dir / "train_log.jsonl",
    )
    (run_paths.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    summary = train(
        config=config,
        model_name=args.model,
        feature_set=args.feature_set,
        run_paths=run_paths,
        device=args.device,
        limit_train=args.limit_train,
        disable_progress=args.no_progress,
        disable_tensorboard=args.no_tensorboard,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
