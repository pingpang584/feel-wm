"""Per-topic message extraction from ROS1 bags.

One streaming pass through the bag fragments; collect every needed topic
into time-sorted numpy arrays. Uses `header.stamp` as the canonical time
when the message carries a header, otherwise the bag receive-time
(`bag_ts`). Returns a dict keyed by topic; values are typed dataclasses
defined below.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rosbags.highlevel import AnyReader


# --- Canonical topic set (labels, auxiliary signals, on-disk outputs) ------

ANCHOR_TOPIC = "/multisense/left/image_rect_color"

TOPICS = {
    ANCHOR_TOPIC,
    "/integrated_to_init",
    "/novatel/imu/data",
    "/span/status",
    "/controls",
    "/wheel_rpm",
    "/shock_pos",
    "/ros_talon/steering_angle",
    "/traversability_cost",
}


# --- Per-topic buffers ------------------------------------------------------


@dataclass
class PoseTwist:
    ts_ns: np.ndarray           # (N,) int64, header.stamp
    pos: np.ndarray             # (N, 3) float64
    quat: np.ndarray            # (N, 4) float64 (x, y, z, w)
    twist_linear: np.ndarray    # (N, 3) float32
    twist_angular: np.ndarray   # (N, 3) float32


@dataclass
class Imu:
    ts_ns: np.ndarray
    accel: np.ndarray           # (N, 3) float32
    gyro: np.ndarray            # (N, 3) float32


@dataclass
class Span:
    ts_ns: np.ndarray
    pitch_deg: np.ndarray       # (N,) float32
    roll_deg: np.ndarray
    azimuth_deg: np.ndarray


@dataclass
class Controls:
    ts_ns: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray


@dataclass
class FourWheel:
    ts_ns: np.ndarray
    fl: np.ndarray
    fr: np.ndarray
    rl: np.ndarray
    rr: np.ndarray


@dataclass
class Scalar:
    ts_ns: np.ndarray
    data: np.ndarray            # (N,) float32


@dataclass
class ImageStream:
    """Store image messages lazily via (bag_index, msg_offset) references.

    For the smoke-test we keep encoded message data in memory because the
    rep sequence (~1720 frames) fits easily. If we go to the full dataset,
    switch this to a streaming writer that decodes+writes PNG per frame.
    """
    ts_ns: np.ndarray
    height: int
    width: int
    encoding: str
    data_bytes: list[np.ndarray]    # len = N, each entry is the raw msg.data


# --- Extraction -------------------------------------------------------------


def _header_ts_ns(msg) -> int | None:
    """Return header.stamp as int ns, or None if the message has no header or stamp is zero."""
    try:
        stamp = msg.header.stamp
    except AttributeError:
        return None
    ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if ns <= 0:
        return None
    return ns


def _sort_buffer(buf: dict) -> dict:
    """Sort every array in `buf` by 'ts_ns'. Returns a new dict."""
    ts = np.asarray(buf["ts_ns"], dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    out = {}
    for k, v in buf.items():
        if k == "ts_ns":
            out[k] = ts[order]
        elif isinstance(v, list):
            out[k] = [v[i] for i in order]
        else:
            arr = np.asarray(v)
            out[k] = arr[order]
    return out


def extract_topics(bag_paths: Iterable[Path]) -> dict:
    """Stream bag_paths once; return dict with typed buffers per canonical topic.

    Keys match the TOPICS constants above, except images which is stored
    under the key 'image'.
    """
    bag_paths = [Path(p) for p in bag_paths]

    # Raw lists per topic
    raw = {t: {"ts_ns": []} for t in TOPICS}
    for t in ("/integrated_to_init",):
        raw[t]["pos"] = []
        raw[t]["quat"] = []
        raw[t]["twist_linear"] = []
        raw[t]["twist_angular"] = []
    for t in ("/novatel/imu/data",):
        raw[t]["accel"] = []
        raw[t]["gyro"] = []
    raw["/span/status"]["pitch_deg"] = []
    raw["/span/status"]["roll_deg"] = []
    raw["/span/status"]["azimuth_deg"] = []
    raw["/controls"]["throttle"] = []
    raw["/controls"]["brake"] = []
    for t in ("/wheel_rpm", "/shock_pos"):
        for f in ("fl", "fr", "rl", "rr"):
            raw[t][f] = []
    raw["/ros_talon/steering_angle"]["data"] = []
    raw["/traversability_cost"]["data"] = []
    raw[ANCHOR_TOPIC]["height"] = None
    raw[ANCHOR_TOPIC]["width"] = None
    raw[ANCHOR_TOPIC]["encoding"] = None
    raw[ANCHOR_TOPIC]["data_bytes"] = []

    with AnyReader(bag_paths) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        for conn, bag_ts, data in reader.messages(connections=conns):
            msg = reader.deserialize(data, conn.msgtype)
            topic = conn.topic

            # Timestamp
            hdr = _header_ts_ns(msg)
            ts_ns = hdr if hdr is not None else int(bag_ts)

            if topic == "/integrated_to_init":
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                tl = msg.twist.twist.linear
                ta = msg.twist.twist.angular
                raw[topic]["ts_ns"].append(ts_ns)
                raw[topic]["pos"].append([p.x, p.y, p.z])
                raw[topic]["quat"].append([q.x, q.y, q.z, q.w])
                raw[topic]["twist_linear"].append([tl.x, tl.y, tl.z])
                raw[topic]["twist_angular"].append([ta.x, ta.y, ta.z])
            elif topic == "/novatel/imu/data":
                a = msg.linear_acceleration
                g = msg.angular_velocity
                raw[topic]["ts_ns"].append(ts_ns)
                raw[topic]["accel"].append([a.x, a.y, a.z])
                raw[topic]["gyro"].append([g.x, g.y, g.z])
            elif topic == "/span/status":
                raw[topic]["ts_ns"].append(ts_ns)
                raw[topic]["pitch_deg"].append(msg.pitch_ins_deg)
                raw[topic]["roll_deg"].append(msg.roll_ins_deg)
                raw[topic]["azimuth_deg"].append(msg.azimuth_ins_deg)
            elif topic == "/controls":
                raw[topic]["ts_ns"].append(ts_ns)
                raw[topic]["throttle"].append(msg.throttle)
                raw[topic]["brake"].append(msg.brake)
            elif topic in ("/wheel_rpm", "/shock_pos"):
                raw[topic]["ts_ns"].append(ts_ns)
                raw[topic]["fl"].append(msg.front_left)
                raw[topic]["fr"].append(msg.front_right)
                raw[topic]["rl"].append(msg.rear_left)
                raw[topic]["rr"].append(msg.rear_right)
            elif topic == "/ros_talon/steering_angle":
                raw[topic]["ts_ns"].append(ts_ns)
                raw[topic]["data"].append(msg.data)
            elif topic == "/traversability_cost":
                raw[topic]["ts_ns"].append(ts_ns)
                raw[topic]["data"].append(msg.data)
            elif topic == ANCHOR_TOPIC:
                raw[topic]["ts_ns"].append(ts_ns)
                if raw[topic]["height"] is None:
                    raw[topic]["height"] = int(msg.height)
                    raw[topic]["width"] = int(msg.width)
                    raw[topic]["encoding"] = msg.encoding
                raw[topic]["data_bytes"].append(np.asarray(msg.data, dtype=np.uint8))

    # Sort each buffer by ts_ns and pack into dataclasses
    out = {}

    it = _sort_buffer(raw["/integrated_to_init"])
    out["/integrated_to_init"] = PoseTwist(
        ts_ns=it["ts_ns"],
        pos=np.asarray(it["pos"], dtype=np.float64).reshape(-1, 3),
        quat=np.asarray(it["quat"], dtype=np.float64).reshape(-1, 4),
        twist_linear=np.asarray(it["twist_linear"], dtype=np.float32).reshape(-1, 3),
        twist_angular=np.asarray(it["twist_angular"], dtype=np.float32).reshape(-1, 3),
    )

    imu = _sort_buffer(raw["/novatel/imu/data"])
    out["/novatel/imu/data"] = Imu(
        ts_ns=imu["ts_ns"],
        accel=np.asarray(imu["accel"], dtype=np.float32).reshape(-1, 3),
        gyro=np.asarray(imu["gyro"], dtype=np.float32).reshape(-1, 3),
    )

    sp = _sort_buffer(raw["/span/status"])
    out["/span/status"] = Span(
        ts_ns=sp["ts_ns"],
        pitch_deg=np.asarray(sp["pitch_deg"], dtype=np.float32),
        roll_deg=np.asarray(sp["roll_deg"], dtype=np.float32),
        azimuth_deg=np.asarray(sp["azimuth_deg"], dtype=np.float32),
    )

    ctl = _sort_buffer(raw["/controls"])
    out["/controls"] = Controls(
        ts_ns=ctl["ts_ns"],
        throttle=np.asarray(ctl["throttle"], dtype=np.float32),
        brake=np.asarray(ctl["brake"], dtype=np.float32),
    )

    for key in ("/wheel_rpm", "/shock_pos"):
        r = _sort_buffer(raw[key])
        out[key] = FourWheel(
            ts_ns=r["ts_ns"],
            fl=np.asarray(r["fl"], dtype=np.float32),
            fr=np.asarray(r["fr"], dtype=np.float32),
            rl=np.asarray(r["rl"], dtype=np.float32),
            rr=np.asarray(r["rr"], dtype=np.float32),
        )

    st = _sort_buffer(raw["/ros_talon/steering_angle"])
    out["/ros_talon/steering_angle"] = Scalar(
        ts_ns=st["ts_ns"],
        data=np.asarray(st["data"], dtype=np.float32),
    )

    tc = _sort_buffer(raw["/traversability_cost"])
    out["/traversability_cost"] = Scalar(
        ts_ns=tc["ts_ns"],
        data=np.asarray(tc["data"], dtype=np.float32),
    )

    # Anchor images — sort by ts_ns, keep as list of uint8 arrays
    img = raw[ANCHOR_TOPIC]
    ts = np.asarray(img["ts_ns"], dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    out[ANCHOR_TOPIC] = ImageStream(
        ts_ns=ts[order],
        height=img["height"] or 0,
        width=img["width"] or 0,
        encoding=img["encoding"] or "",
        data_bytes=[img["data_bytes"][i] for i in order],
    )

    return out
