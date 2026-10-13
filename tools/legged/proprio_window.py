"""The 53-column causal proprio window builder — ONE definition of the encoder input.

The legged schema's encoder input is a causal window ``(50, 53)`` sampled on a 100 Hz
grid ending at the anchor time ``t``. ``tools/legged/extract_isaac_anymal.py`` builds it
here and writes it into ``labels_legged.npz[proprio_window]``, which is what the WM is
TRAINED on. Any other consumer that needs the window at run time (e.g. feeding a live
robot's sensor buffers to the same checkpoint) must build it with this module too.

A silent drift between two builders would not crash anything: the model would simply be
fed a window from a different distribution than it was trained on, and every number
after that would be uninterpretable. So the assembly (column order, staging, causal
resampling) lives in one place and callers import it.

Column layout (53) — the order is the schema, do not reorder:

    [ 0: 3)  acc x,y,z          IMU clock
    [ 3: 6)  gyro x,y,z         IMU clock
    [ 6: 9)  twist vx, vy, wz   odom clock, BODY frame
    [ 9:10)  pitch (rad)        odom clock
    [10:11)  roll  (rad)        odom clock
    [11:13)  cmd v, w           cmd clock
    [13:25)  joint position  12 } joint clock, CANONICAL per-leg order
    [25:37)  joint velocity  12 }   (LF/RF/LH/RH x HAA/HFE/KFE — `urdf_fk.CANONICAL_JOINTS`),
    [37:49)  joint torque    12 }   permuted from the bag's Isaac TREE order
    [49:53)  foot Fz          4   joint clock, feet LF, RF, LH, RH

Resampling is HOLD-LAST (zero-order hold on the last sample at or before each grid
point), never interpolation: the window must be causal at every one of its 50 points,
and a linear interpolant at grid point g would read a sample from after g.

Sources may arrive on different clocks at different rates (IMU 200 Hz, joints+feet
100 Hz, odom+cmd 50 Hz) — that is exactly what the collector records, so the hold-last
grid is where they get aligned. Foot Fz is required to share the JOINT clock, which is
what the extractor gives it (it interpolates Fz onto ``jt_ts`` when reading the bag);
a live source must likewise sample joints and feet on the same clock.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

ENC_DT = 0.01          # 100 Hz window grid
T_WIN_N = 50           # causal window samples (0.5 s span, ending at t)
D_RAW = 53             # window width — must equal the checkpoint's model.d_raw

# (name, width) in column order — the documented layout above, machine-readable so a
# test can assert the widths sum to D_RAW instead of trusting the comment.
COLUMN_GROUPS = (
    ("acc", 3), ("gyro", 3), ("twist", 3), ("pitch", 1), ("roll", 1), ("cmd", 2),
    ("joint_pos", 12), ("joint_vel", 12), ("joint_tau", 12), ("foot_fz", 4),
)


def quat_pitch_roll(q_xyzw):
    """(N,4) xyzw quaternions → (pitch, roll) arrays, both radians.

    Column 9 is pitch and column 10 is roll (not the intuitive roll-then-pitch); the
    order is the schema's, and both callers must read it from here.
    """
    e = Rotation.from_quat(q_xyzw).as_euler("xyz")      # roll, pitch, yaw
    return e[:, 1], e[:, 0]


def joint_permutation(joint_names) -> list[int]:
    """Isaac TREE order → canonical per-leg order (`urdf_fk.CANONICAL_JOINTS`).

    The collector records `/joint_states` in the articulation's own tree order
    (HAA x4 | HFE x4 | KFE x4); the schema is per-leg. The permutation is a property of
    the recorded name list, so it is derived from it rather than hard-coded — a robot
    whose articulation enumerates differently still lands on the same 12 columns.
    """
    from urdf_fk import CANONICAL_JOINTS

    names = list(joint_names)
    missing = [j for j in CANONICAL_JOINTS if j not in names]
    if missing:
        raise ValueError(f"joint name list is missing canonical joints {missing}; got {names}")
    return [names.index(j) for j in CANONICAL_JOINTS]


def hold_last(src_ts, src_val, grid):
    """Zero-order hold of ``src_val`` (stamped ``src_ts``, ascending) onto ``grid``.

    Empty source → all-NaN of the source's own trailing shape, so a missing topic shows
    up as NaN in the window (and, offline, invalidates the anchor) instead of as zeros
    that look like real data.
    """
    if len(src_ts) == 0:
        return np.full((len(grid),) + src_val.shape[1:], np.nan, np.float32)
    idx = np.clip(np.searchsorted(src_ts, grid, side="right") - 1, 0, len(src_ts) - 1)
    return src_val[idx]


def causal_window(src_ts, src_val, t):
    """``(T_WIN_N, ...)`` hold-last samples on the grid ``[t-0.49 … t]`` at 100 Hz."""
    grid = t - ENC_DT * np.arange(T_WIN_N - 1, -1, -1)
    return hold_last(src_ts, src_val, grid)


@dataclass(frozen=True)
class WindowSources:
    """Staged, column-ready signal blocks + their clocks.

    Built once per episode (or once per observation for a live source). The staging (the two
    concatenations and the pitch/roll extraction) is part of the definition, so it lives
    here too — a caller that stacked ``[jvel, jpos, jtau]`` in its own order would
    otherwise produce a window that is the right shape and the wrong thing.
    """

    imu_ts: np.ndarray      # (N,)      IMU clock
    imu: np.ndarray         # (N, 6)    acc xyz | gyro xyz
    od_ts: np.ndarray       # (M,)      odom clock
    twist: np.ndarray       # (M, 3)    vx, vy, wz (body frame)
    pitch: np.ndarray       # (M, 1)
    roll: np.ndarray        # (M, 1)
    cm_ts: np.ndarray       # (L,)      cmd clock
    cmd: np.ndarray         # (L, 2)    v, w
    jt_ts: np.ndarray       # (K,)      joint clock (feet share it)
    joints: np.ndarray      # (K, 36)   jpos12 | jvel12 | jtau12, CANONICAL order
    fz: np.ndarray          # (K, 4)    LF, RF, LH, RH


def window_sources(*, imu_ts, acc, gyro, od_ts, lin, ang, pitch, roll,
                   cm_ts, cmd, jt_ts, jpos, jvel, jtau, fz) -> WindowSources:
    """Stage raw per-topic arrays into :class:`WindowSources`.

    ``lin``/``ang`` are the odometry twist in the BODY frame (the bag contract), and only
    vx, vy, wz reach the window. ``jpos``/``jvel``/``jtau`` must already be in canonical
    per-leg order (:func:`joint_permutation`), and ``fz`` must be on the joint clock.
    """
    lin = np.asarray(lin)
    ang = np.asarray(ang)
    twist = np.concatenate([lin[:, [0, 1]], ang[:, 2:3]], axis=1)
    return WindowSources(
        imu_ts=np.asarray(imu_ts), imu=np.concatenate([np.asarray(acc), np.asarray(gyro)], axis=1),
        od_ts=np.asarray(od_ts), twist=twist,
        pitch=np.asarray(pitch)[:, None], roll=np.asarray(roll)[:, None],
        cm_ts=np.asarray(cm_ts), cmd=np.asarray(cmd),
        jt_ts=np.asarray(jt_ts),
        joints=np.concatenate([np.asarray(jpos), np.asarray(jvel), np.asarray(jtau)], axis=1),
        fz=np.asarray(fz),
    )


def build_window(src: WindowSources, t: float) -> np.ndarray:
    """``(50, 53)`` float32 causal window ending at ``t``."""
    w_cmd = (causal_window(src.cm_ts, src.cmd, t) if len(src.cm_ts)
             else np.full((T_WIN_N, 2), np.nan, np.float32))
    return np.concatenate([
        causal_window(src.imu_ts, src.imu, t),          # acc3 + gyro3
        causal_window(src.od_ts, src.twist, t),         # vx, vy, wz
        causal_window(src.od_ts, src.pitch, t),
        causal_window(src.od_ts, src.roll, t),
        w_cmd,                                          # cmd v, w
        causal_window(src.jt_ts, src.joints, t),        # jpos12 | jvel12 | jtau12
        causal_window(src.jt_ts, src.fz, t),            # foot Fz 4
    ], axis=1).astype(np.float32)


def build_windows(src: WindowSources, grid) -> np.ndarray:
    """``(N, 50, 53)`` float32 windows, one per anchor time in ``grid``."""
    out = np.full((len(grid), T_WIN_N, D_RAW), np.nan, np.float32)
    for i, t in enumerate(grid):
        out[i] = build_window(src, float(t))
    return out
