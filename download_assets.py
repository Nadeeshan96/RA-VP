#!/usr/bin/env python3
"""Download + verify + extract the external asset bundle (data, weights, features).

The code repo ships only code + small precomputed results. The large files
(processed CSVs, forecaster + window/baseline caches, RA-VP features) live in a
single tarball hosted off-repo.

Usage:
  python download_assets.py                      # download per the manifest URL
  python download_assets.py --bundle path.tgz    # use an already-downloaded tarball
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "assets_manifest.json"


def _drive_id(url: str):
    """Extract a Google Drive file id from a share URL, else None."""
    if "drive.google.com" not in url:
        return None
    m = re.search(r"/d/([A-Za-z0-9_-]+)", url) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None, help="path to an already-downloaded tarball")
    a = ap.parse_args()
    m = json.loads(MANIFEST.read_text())["bundle"]
    dest = ROOT / m["filename"]

    if a.bundle:
        dest = Path(a.bundle)
    elif not dest.exists():
        url = m.get("url", "")
        gid = m.get("gdrive_id") or _drive_id(url)
        if gid:
            try:
                import gdown
            except ImportError:
                sys.exit("This bundle is on Google Drive; please `pip install gdown` "
                         "(it handles Drive's large-file confirmation page), then re-run.")
            print(f"[download] Google Drive id={gid}\n        -> {dest}")
            gdown.download(id=gid, output=str(dest), quiet=False)
        elif url and not url.startswith("REPLACE"):
            print(f"[download] {url}\n        -> {dest}")
            urllib.request.urlretrieve(url, dest)
        else:
            sys.exit("assets_manifest.json has no usable URL. Set 'url' (or 'gdrive_id'), "
                     "or pass --bundle <path>.")

    print(f"[verify] sha256 of {dest.name} ...")
    got = sha256(dest)
    if got != m["sha256"]:
        sys.exit(f"checksum mismatch!\n  expected {m['sha256']}\n  got      {got}")
    print("  ok")

    out = ROOT / m["extract_to"]
    print(f"[extract] -> {out}")
    with tarfile.open(dest, "r:gz") as t:
        t.extractall(out)
    print("[done] assets ready under", ROOT / "assets")


if __name__ == "__main__":
    main()
