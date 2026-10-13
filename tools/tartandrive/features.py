"""5-dim proprio feature label [roughness, slip_ratio, angular_instability, ke_rate,
slope_power], 14-dim encoder-input window, auxiliary susp_rms. Pure numpy. Window specs:

- Label features: centered [t − 0.25 s, t + 0.25 s].
- Encoder-input window: causal [t − 0.5 s, t], resampled to 100 Hz (T_win = 50).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .extract import Imu, PoseTwist, Span, Controls, FourWheel, Scalar


R_TIRE_M = 0.3175
TIRE_SCALE = (2.0 * np.pi / 60.0) * R_TIRE_M  # rpm → m/s
RPM_OUTLIER_THRESHOLD = 400.0
SLIP_EPS = 0.1

T_WIN_N = 50           # encoder-input window samples
T_WIN_SEC = 0.5
LABEL_HALFWIN_NS = int(0.25 * 1e9)
CAUSAL_WIN_NS = int(0.5 * 1e9)
ENC_SAMPLE_DT_NS = int(10e6)  # 100 Hz stride = 10 ms per sample


# --- Small utilities --------------------------------------------------------


def window_indices(src_ts: np.ndarray, t0_ns: int, t1_ns: int) -> np.ndarray:
    """Inclusive [t0, t1] slice indices into a sorted-ascending src_ts array."""
    lo = np.searchsorted(src_ts, t0_ns, side="left")
    hi = np.searchsorted(src_ts, t1_ns, side="right")
    return np.arange(lo, hi, dtype=np.int64)


def nearest_idx(src_ts: np.ndarray, t_ns: int) -> tuple[int, int]:
    """Return (idx, abs_dt_ns) for nearest neighbor in sorted src_ts."""
    if len(src_ts) == 0:
        return -1, 10**18
    right = int(np.searchsorted(src_ts, t_ns))
    left = max(right - 1, 0)
    right = min(right, len(src_ts) - 1)
    dl = abs(int(src_ts[left]) - t_ns)
    dr = abs(int(src_ts[right]) - t_ns)
    if dl <= dr:
        return left, dl
    return right, dr


def hold_last_resample(src_ts: np.ndarray, src_val: np.ndarray,
                       grid_ts: np.ndarray) -> np.ndarray:
    """Zero-order hold: for each grid_ts, use the latest src sample with ts <= grid_ts.

    If no src sample precedes a given grid point, the first src sample is used.
    """
    if len(src_ts) == 0:
        return np.full_like(grid_ts, np.nan, dtype=np.float32)
    # Index of the largest src_ts that is <= grid_ts_i, per grid point.
    idx = np.searchsorted(src_ts, grid_ts, side="right") - 1
    idx = np.clip(idx, 0, len(src_ts) - 1)
    return src_val[idx]


# --- 5-dim label features --------------------------------------------------


def roughness(imu: Imu, t_ns: int) -> tuple[float, int]:
    """std of a_z − mean(a_z) over centered 0.5 s window. Returns (value, n_samples)."""
    idx = window_indices(imu.ts_ns, t_ns - LABEL_HALFWIN_NS, t_ns + LABEL_HALFWIN_NS)
    if len(idx) < 3:
        return np.nan, len(idx)
    az = imu.accel[idx, 2]
    return float(np.std(az - np.mean(az), ddof=0)), len(idx)


def slip_ratio(pose: PoseTwist, wheel: FourWheel | None, t_ns: int) -> tuple[float, int, int]:
    """Mean slip over centered 0.5 s window.
    Returns (value, n_samples_used, n_samples_dropped).
    """
    if wheel is None or len(wheel.ts_ns) == 0:
        return np.nan, 0, 0
    if pose is None or len(pose.ts_ns) == 0:
        return np.nan, 0, 0
    widx = window_indices(wheel.ts_ns, t_ns - LABEL_HALFWIN_NS, t_ns + LABEL_HALFWIN_NS)
    if len(widx) < 3:
        return np.nan, len(widx), 0
    wfl = wheel.fl[widx]
    wfr = wheel.fr[widx]
    wrl = wheel.rl[widx]
    wrr = wheel.rr[widx]
    ok = ((wfl <= RPM_OUTLIER_THRESHOLD) & (wfr <= RPM_OUTLIER_THRESHOLD)
          & (wrl <= RPM_OUTLIER_THRESHOLD) & (wrr <= RPM_OUTLIER_THRESHOLD))
    n_drop = int((~ok).sum())
    if ok.sum() < 3:
        return np.nan, int(ok.sum()), n_drop
    v_wheel = TIRE_SCALE * 0.25 * (wfl[ok] + wfr[ok] + wrl[ok] + wrr[ok])
    # Ground speed sampled at the same wheel timestamps
    pts = wheel.ts_ns[widx][ok]
    vgrid_x = np.interp(pts, pose.ts_ns, pose.twist_linear[:, 0])
    vgrid_y = np.interp(pts, pose.ts_ns, pose.twist_linear[:, 1])
    v_ground = np.hypot(vgrid_x, vgrid_y)
    # Symmetric denominator: clamps |slip| ≤ 1 regardless of which speed is
    # near zero. Avoids blow-up during braking / coasting (v_wheel → 0 with
    # v_ground > 0) and spinning-on-the-spot (v_ground → 0 with v_wheel > 0).
    denom = np.maximum.reduce([v_wheel, v_ground, np.full_like(v_wheel, SLIP_EPS)])
    slip = (v_wheel - v_ground) / denom
    return float(np.mean(slip)), int(ok.sum()), n_drop


def angular_instability(imu: Imu, t_ns: int) -> float:
    """sqrt(var(roll_rate) + var(pitch_rate)) over centered 0.5 s window."""
    idx = window_indices(imu.ts_ns, t_ns - LABEL_HALFWIN_NS, t_ns + LABEL_HALFWIN_NS)
    if len(idx) < 3:
        return np.nan
    roll_rate = imu.gyro[idx, 0]
    pitch_rate = imu.gyro[idx, 1]
    return float(np.sqrt(np.var(roll_rate) + np.var(pitch_rate)))


def ke_rate(pose: PoseTwist, t_ns: int) -> float:
    """mean(v · dv/dt) over centered 0.5 s window with mild LPF on v."""
    if pose is None or len(pose.ts_ns) == 0:
        return np.nan
    idx = window_indices(pose.ts_ns, t_ns - LABEL_HALFWIN_NS, t_ns + LABEL_HALFWIN_NS)
    if len(idx) < 7:
        return np.nan
    v = np.hypot(pose.twist_linear[idx, 0], pose.twist_linear[idx, 1])
    wl = min(len(v), 5)
    if wl % 2 == 0:
        wl -= 1
    if wl < 3:
        v_smooth = v
    else:
        v_smooth = savgol_filter(v, window_length=wl, polyorder=min(2, wl - 1))
    dv = np.gradient(v_smooth)
    # dt in seconds
    dt = np.diff(pose.ts_ns[idx]).astype(np.float64) / 1e9
    dt_per_sample = np.concatenate([[dt[0]], dt])
    dv_dt = dv / np.maximum(dt_per_sample, 1e-6)
    return float(np.mean(v * dv_dt))


def slope_power(pose: PoseTwist, span: Span, t_ns: int) -> float:
    """mean(v · sin(pitch)) over centered 0.5 s window."""
    if pose is None or len(pose.ts_ns) == 0:
        return np.nan
    idx = window_indices(pose.ts_ns, t_ns - LABEL_HALFWIN_NS, t_ns + LABEL_HALFWIN_NS)
    if len(idx) < 3 or len(span.ts_ns) == 0:
        return np.nan
    v = np.hypot(pose.twist_linear[idx, 0], pose.twist_linear[idx, 1])
    pitch_rad = np.deg2rad(np.interp(pose.ts_ns[idx], span.ts_ns, span.pitch_deg))
    return float(np.mean(v * np.sin(pitch_rad)))


def proprio_feature_5d(
    pose: PoseTwist, imu: Imu, span: Span, wheel: FourWheel | None, t_ns: int
) -> tuple[np.ndarray, dict]:
    """Return (f_prop[5], diagnostics)."""
    r, _ = roughness(imu, t_ns)
    s, n_ok, n_drop = slip_ratio(pose, wheel, t_ns)
    ai = angular_instability(imu, t_ns)
    ke = ke_rate(pose, t_ns)
    sp = slope_power(pose, span, t_ns)
    arr = np.array([r, s, ai, ke, sp], dtype=np.float32)
    diag = {"slip_n_ok": n_ok, "slip_n_drop": n_drop}
    return arr, diag


# --- 14-dim encoder-input window -------------------------------------------


def encoder_input_window(
    pose: PoseTwist,
    imu: Imu,
    span: Span,
    controls: Controls | None,
    steer: Scalar | None,
    t_ns: int,
) -> np.ndarray:
    """Shape (T_WIN_N, 14). Causal [t - 0.5s, t], resampled to 100 Hz."""
    # Build causal grid: t - (T_WIN_N-1)*dt, ..., t
    grid = t_ns - np.arange(T_WIN_N - 1, -1, -1, dtype=np.int64) * ENC_SAMPLE_DT_NS
    out = np.full((T_WIN_N, 14), np.nan, dtype=np.float32)

    # Columns 0-5: IMU accel (3) + gyro (3), hold-last
    if len(imu.ts_ns) > 0:
        for c in range(3):
            out[:, c] = hold_last_resample(imu.ts_ns, imu.accel[:, c], grid)
        for c in range(3):
            out[:, 3 + c] = hold_last_resample(imu.ts_ns, imu.gyro[:, c], grid)

    # Columns 6-8: body twist from /integrated_to_init
    if len(pose.ts_ns) > 0:
        out[:, 6] = hold_last_resample(pose.ts_ns, pose.twist_linear[:, 0], grid)
        out[:, 7] = hold_last_resample(pose.ts_ns, pose.twist_linear[:, 1], grid)
        out[:, 8] = hold_last_resample(pose.ts_ns, pose.twist_angular[:, 2], grid)

    # Columns 9-10: pitch, roll (rad)
    if len(span.ts_ns) > 0:
        out[:, 9] = np.deg2rad(hold_last_resample(span.ts_ns, span.pitch_deg, grid))
        out[:, 10] = np.deg2rad(hold_last_resample(span.ts_ns, span.roll_deg, grid))

    # Columns 11-12: throttle, brake (raw)
    if controls is not None and len(controls.ts_ns) > 0:
        out[:, 11] = hold_last_resample(controls.ts_ns, controls.throttle, grid)
        out[:, 12] = hold_last_resample(controls.ts_ns, controls.brake, grid)

    # Column 13: commanded steering
    if steer is not None and len(steer.ts_ns) > 0:
        out[:, 13] = hold_last_resample(steer.ts_ns, steer.data, grid)

    return out


# --- Auxiliary susp_rms ----------------------------------------------------


def susp_rms(shock: FourWheel | None, t_ns: int) -> float:
    """Rear-only suspension work-rate proxy, per-pair dt (auxiliary, not a label)."""
    if shock is None or len(shock.ts_ns) < 3:
        return np.nan
    idx = window_indices(shock.ts_ns, t_ns - LABEL_HALFWIN_NS, t_ns + LABEL_HALFWIN_NS)
    if len(idx) < 3:
        return np.nan
    ts = shock.ts_ns[idx].astype(np.float64) / 1e9
    rl = shock.rl[idx]
    rr = shock.rr[idx]
    dt = np.diff(ts)
    if np.any(dt <= 0):
        return np.nan
    v_rl = np.diff(rl) / dt
    v_rr = np.diff(rr) / dt
    return float(np.sqrt(np.mean(v_rl ** 2 + v_rr ** 2) / 2.0))
