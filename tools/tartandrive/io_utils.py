"""On-disk writers for a processed sequence: labels npz, meta.json, sample CSV, images."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .extract import ImageStream


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def write_sample_csv(path: Path, arrays: dict[str, np.ndarray], n: int = 200) -> None:
    """Flat CSV of first `n` anchors. pose flattened as pose_00..pose_33;
    proprio_window written as proprio_window_shape only (too wide for CSV)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(n, len(arrays["anchor_idx"]))
    cols = []
    rows_dict = {}
    for k, v in arrays.items():
        if k == "proprio_window":
            # too wide to dump inline; note shape only
            rows_dict[f"{k}_shape"] = [str(list(v[i].shape)) for i in range(n)]
            cols.append(f"{k}_shape")
            continue
        if v.ndim == 1:
            rows_dict[k] = v[:n].tolist()
            cols.append(k)
        elif v.ndim == 2:
            for i in range(v.shape[1]):
                key = f"{k}_{i}"
                rows_dict[key] = v[:n, i].tolist()
                cols.append(key)
        elif v.ndim == 3:
            # e.g. pose (N,4,4) → pose_ij
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    key = f"{k}_{i}{j}"
                    rows_dict[key] = v[:n, i, j].tolist()
                    cols.append(key)
    with path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for row_i in range(n):
            f.write(",".join(f"{rows_dict[c][row_i]!r}" if isinstance(rows_dict[c][row_i], str)
                             else f"{rows_dict[c][row_i]}"
                             for c in cols) + "\n")


def write_images(out_dir: Path, images: ImageStream) -> int:
    """Decode each bgr8 message, write to out_dir/<idx>.png. Returns number written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = images.height, images.width
    enc = images.encoding
    n = len(images.ts_ns)
    for i in range(n):
        data = images.data_bytes[i]
        if enc == "bgr8":
            arr = data.reshape(h, w, 3)[:, :, ::-1]   # BGR → RGB
        elif enc == "rgb8":
            arr = data.reshape(h, w, 3)
        elif enc == "mono8":
            arr = data.reshape(h, w)
        else:
            raise ValueError(f"Unsupported encoding: {enc}")
        Image.fromarray(arr).save(out_dir / f"{i}.png", compress_level=6)
    return n
