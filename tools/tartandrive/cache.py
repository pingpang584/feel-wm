"""Intermediate cache layer — save the output of `extract.extract_topics()` to
`extracted.npz` so feature recomputation can skip the expensive bag I/O.

Images are NOT cached here (they live as PNGs in `images/left_color/`).
Only their timestamps and meta (height/width/encoding) are saved."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .extract import (
    ANCHOR_TOPIC,
    Controls,
    FourWheel,
    ImageStream,
    Imu,
    PoseTwist,
    Scalar,
    Span,
)

CACHE_VERSION = 1


def save_extracted(path: Path, data: dict) -> None:
    """Dump the dict returned by extract_topics() to a compressed npz.
    Image bytes are dropped; only ts/height/width/encoding are preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pose = data["/integrated_to_init"]
    imu = data["/novatel/imu/data"]
    span = data["/span/status"]
    controls = data.get("/controls")
    wheel = data.get("/wheel_rpm")
    shock = data.get("/shock_pos")
    steer = data.get("/ros_talon/steering_angle")
    trav = data.get("/traversability_cost")
    images = data[ANCHOR_TOPIC]

    out = {
        "__cache_version__": np.array(CACHE_VERSION, dtype=np.int32),
        # pose
        "pose_ts_ns": pose.ts_ns.astype(np.int64),
        "pose_pos": pose.pos.astype(np.float64),
        "pose_quat": pose.quat.astype(np.float64),
        "pose_twist_linear": pose.twist_linear.astype(np.float32),
        "pose_twist_angular": pose.twist_angular.astype(np.float32),
        # imu
        "imu_ts_ns": imu.ts_ns.astype(np.int64),
        "imu_accel": imu.accel.astype(np.float32),
        "imu_gyro": imu.gyro.astype(np.float32),
        # span
        "span_ts_ns": span.ts_ns.astype(np.int64),
        "span_pitch_deg": span.pitch_deg.astype(np.float32),
        "span_roll_deg": span.roll_deg.astype(np.float32),
        "span_azimuth_deg": span.azimuth_deg.astype(np.float32),
        # anchor (images)
        "anchor_ts_ns": images.ts_ns.astype(np.int64),
        "image_height": np.array(images.height, dtype=np.int32),
        "image_width": np.array(images.width, dtype=np.int32),
        "image_encoding": np.array(images.encoding, dtype="<U16"),
    }
    if controls is not None and len(controls.ts_ns) > 0:
        out["controls_ts_ns"] = controls.ts_ns.astype(np.int64)
        out["controls_throttle"] = controls.throttle.astype(np.float32)
        out["controls_brake"] = controls.brake.astype(np.float32)
    for key, entry in (("wheel_rpm", wheel), ("shock", shock)):
        if entry is None or len(entry.ts_ns) == 0:
            continue
        out[f"{key}_ts_ns"] = entry.ts_ns.astype(np.int64)
        out[f"{key}_fl"] = entry.fl.astype(np.float32)
        out[f"{key}_fr"] = entry.fr.astype(np.float32)
        out[f"{key}_rl"] = entry.rl.astype(np.float32)
        out[f"{key}_rr"] = entry.rr.astype(np.float32)
    if steer is not None and len(steer.ts_ns) > 0:
        out["steer_ts_ns"] = steer.ts_ns.astype(np.int64)
        out["steer_data"] = steer.data.astype(np.float32)
    if trav is not None and len(trav.ts_ns) > 0:
        out["trav_ts_ns"] = trav.ts_ns.astype(np.int64)
        out["trav_data"] = trav.data.astype(np.float32)

    np.savez_compressed(path, **out)


def load_extracted(path: Path) -> dict:
    """Load extracted.npz and rebuild a dict matching extract_topics() shape.
    ImageStream.data_bytes is None (images live on disk as PNGs)."""
    z = np.load(path, allow_pickle=False)
    ver = int(z["__cache_version__"])
    if ver != CACHE_VERSION:
        raise RuntimeError(
            f"{path}: cache version {ver} != current {CACHE_VERSION}; rebuild."
        )
    out: dict = {}
    out["/integrated_to_init"] = PoseTwist(
        ts_ns=z["pose_ts_ns"],
        pos=z["pose_pos"],
        quat=z["pose_quat"],
        twist_linear=z["pose_twist_linear"],
        twist_angular=z["pose_twist_angular"],
    )
    out["/novatel/imu/data"] = Imu(
        ts_ns=z["imu_ts_ns"],
        accel=z["imu_accel"],
        gyro=z["imu_gyro"],
    )
    out["/span/status"] = Span(
        ts_ns=z["span_ts_ns"],
        pitch_deg=z["span_pitch_deg"],
        roll_deg=z["span_roll_deg"],
        azimuth_deg=z["span_azimuth_deg"],
    )
    if "controls_ts_ns" in z.files:
        out["/controls"] = Controls(
            ts_ns=z["controls_ts_ns"],
            throttle=z["controls_throttle"],
            brake=z["controls_brake"],
        )
    for key, topic in (("wheel_rpm", "/wheel_rpm"), ("shock", "/shock_pos")):
        if f"{key}_ts_ns" in z.files:
            out[topic] = FourWheel(
                ts_ns=z[f"{key}_ts_ns"],
                fl=z[f"{key}_fl"],
                fr=z[f"{key}_fr"],
                rl=z[f"{key}_rl"],
                rr=z[f"{key}_rr"],
            )
    if "steer_ts_ns" in z.files:
        out["/ros_talon/steering_angle"] = Scalar(
            ts_ns=z["steer_ts_ns"],
            data=z["steer_data"],
        )
    if "trav_ts_ns" in z.files:
        out["/traversability_cost"] = Scalar(
            ts_ns=z["trav_ts_ns"],
            data=z["trav_data"],
        )
    out[ANCHOR_TOPIC] = ImageStream(
        ts_ns=z["anchor_ts_ns"],
        height=int(z["image_height"]),
        width=int(z["image_width"]),
        encoding=str(z["image_encoding"]),
        data_bytes=[],  # NOT loaded; images already on disk
    )
    return out
