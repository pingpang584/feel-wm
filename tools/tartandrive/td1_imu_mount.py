"""IMU mount rotation for TartanDrive 1.0.

TD2's `/novatel/imu/data` sits body-aligned: its mean acceleration is
`[-0.01, -0.03, +1.00] * g`, so the roughness label can read straight off
`linear_acceleration.z` and the angular-instability label can read roll/pitch rates
off gyro x/y. TD1 has no NovAtel; its only IMU is `/multisense/imu/imu_data`, whose mean
acceleration is `[-0.97, -0.02, -0.24] * g` — gravity lies on **-x**, and the
horizontal axes are rotated by an unknown amount relative to the vehicle.

Feeding that frame to the shared formulas would silently compute roughness from a
horizontal axis. This module solves the constant IMU→body rotation from data, so
that the TD1 extractor can hand `features.py` an IMU in the same convention TD2
uses and the label formulas stay literally shared.

## How the rotation is identified

**Vertical (2 DOF) — gravity.** Near-stationary-acceleration samples
(`| ||a|| - g | < 0.6`) measure gravity in the IMU frame. The same direction is
known in the body frame from the odometry attitude. Kabsch over that correspondence
pins the two tilt degrees of freedom.

**Azimuth (1 DOF) — gyro vs INS attitude rate.** Gravity alone leaves rotation
about the vertical unidentified. It is recovered by matching the horizontal gyro
components to `d(roll)/dt, d(pitch)/dt` differentiated from the odometry attitude.
Two other candidates were measured and rejected:

| method | conditioning | cross-bag azimuth spread |
|---|---|---|
| centripetal `a_lat ~ v * yaw_rate` | ‖(c,s)‖ = 0.08–0.16, r = 0.30–0.43 | 71° — rejected |
| **gyro vs attitude rate** | singular values 0.98–1.11 (a near-rotation), r up to 0.98 | ±12° |

The gyro fit is well conditioned exactly when the run has attitude variation: on
the rough, slow probe bag (`20210826_51`, pitch span 13°) it reaches r = 0.96/0.98.
The flat, fast bags are the ones that disagree, which is why the azimuth is taken
from the best-conditioned bag rather than averaged.

Composition is the subtle part: `Rz(-phi)` removes the measured azimuth, and
transposing it applies `Rz(+phi)` instead. Getting that backwards leaves `2*phi` of
error that no gravity check can see — see MOUNT_PROVENANCE below.

## Why one constant for the whole corpus

The sensor is rigidly mounted, so there is a single true rotation. Solving per bag
would let a nearly-flat run's ill-conditioned azimuth (−12.5° on `20210903_26`)
put TD1 sequences into frames up to ~24° apart — inconsistency *inside* the
training set, which is worse than a constant offset shared by all of it. The
constant below is therefore frozen and applied to every TD1 bag.

Only 4 of the 14 encoder columns depend on the azimuth at all: **both label
channels are azimuth-invariant** (`std(a·ĝ)` uses the vertical only, and the
roll+pitch rate variance sum is invariant to rotation within the horizontal
plane), so the labels themselves do not depend on the azimuth part of this solve.

Re-estimate from bags:

    python -m tools.tartandrive.td1_imu_mount <bag> [<bag> ...]

Validate against an already-built corpus (no bags needed — the cached
`extracted.npz` carries the rotated IMU and the INS attitude):

    python -m tools.tartandrive.td1_check_imu_mount
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

IMU_TOPIC = "/multisense/imu/imu_data"
ODOM_TOPIC = "/odometry/filtered_odom"

GRAVITY = 9.81
QUIET_TOL = 0.3          # |‖a‖ − g| below this ≈ gravity-only sample
QUIET_GYRO_RADS = 0.10   # and the vehicle must not be rotating
GYRO_SMOOTH_N = 9        # IMU runs at ~200-400 Hz, attitude at 50 Hz
MIN_QUIET = 200

# Frozen IMU→body rotation, body axes in the TD1 convention (forward +y, right +x,
# up +z; established by probing TD1 bags).
#
# Pooled gravity solve over 3910 quiet samples (median residual 7.23 deg, which is
# dominated by genuine vehicle acceleration during off-road driving), azimuth from
# the best-conditioned bag, then re-orthonormalised.
#
# Verified twice over. On the three probe bags, against two quantities the fit never
# saw: body yaw rate vs the published odometry yaw rate (r = +0.96 / +0.98 / +0.99,
# scale 1.00-1.04) and mean body acceleration vs the value predicted by each run's
# mean attitude (within 0.2 m/s^2). Then corpus-wide once all 574 sequences existed,
# from their cached artifacts: a Kabsch fit of measured gravity against the INS
# attitude wants a 13.9 deg correction but that correction improves the median
# gravity-direction error by 0.08 deg (7.08 -> 7.00), i.e. the 7 deg residual is
# vehicle acceleration and sensor noise, not misalignment.
#
# The AZIMUTH is what the corpus caught. The first frozen value composed the
# correction as `R_grav @ R_az.T` with R_az = Rz(-phi), which applies Rz(+phi) and
# so ADDS the measured azimuth instead of removing it, leaving 2*phi = 13.98 deg of
# error. The corpus-wide gyro-vs-attitude-rate fit measured that residual
# independently as 14.05 deg over 1.8M samples from all 22 archives, agreeing with
# the analytic 13.98 to 0.07 deg. Ruled out as yaw-rate coupling first: adding the
# yaw axis to the fit moves it by 0.01 deg and gating |gyro_z| from 0.30 to 0.04
# rad/s leaves it at 13.4-14.0 deg.
MOUNT_ROTATION = np.array([
    [+0.005703, -0.249690, -0.968309],
    [-0.999942, -0.010324, -0.003227],
    [-0.009191, +0.968271, -0.249735],
], dtype=np.float64)

MOUNT_PROVENANCE = "probe-gravity-pooled+corpus-azimuth-2026-08-09"


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def attitude_rad(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pitch, roll in the SPAN sense, TD1 convention (forward +y, right +x)."""
    pitch = np.arcsin(np.clip(R[:, 2, 1], -1.0, 1.0))
    roll = -np.arctan2(R[:, 2, 0], R[:, 2, 2])
    return pitch, roll


def _read(bag_path: Path):
    it, acc, gyr, ot, quat, yawrate = [], [], [], [], [], []
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic in (IMU_TOPIC, ODOM_TOPIC)]
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if conn.topic == IMU_TOPIC:
                a, g = m.linear_acceleration, m.angular_velocity
                it.append(t)
                acc.append((a.x, a.y, a.z))
                gyr.append((g.x, g.y, g.z))
            else:
                o = m.pose.pose.orientation
                ot.append(t)
                quat.append((o.x, o.y, o.z, o.w))
                yawrate.append(m.twist.twist.angular.z)
    it = np.asarray(it)
    order = np.argsort(it, kind="stable")     # 5.6% of stamps arrive out of order
    return (it[order], np.asarray(acc)[order], np.asarray(gyr)[order],
            np.asarray(ot), np.asarray(quat), np.asarray(yawrate))


def solve_bag(bag_path: Path) -> dict:
    """Return {'R': mount rotation, ...diagnostics} for one bag."""
    it, acc, gyr, ot, quat, _ = _read(bag_path)
    if it.size == 0 or ot.size == 0:
        return {"bag": Path(bag_path).name, "error": "missing IMU or odometry"}

    R = quat_to_matrix(quat)
    up_body = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, 1.0]))   # Rᵀ @ z_world
    up_imu_t = np.stack([np.interp(it, ot, up_body[:, k]) for k in range(3)], -1)
    up_imu_t /= np.linalg.norm(up_imu_t, axis=1, keepdims=True)

    quiet = ((np.abs(np.linalg.norm(acc, axis=1) - GRAVITY) < QUIET_TOL)
             & (np.linalg.norm(gyr, axis=1) < QUIET_GYRO_RADS))
    if quiet.sum() < MIN_QUIET:
        return {"bag": Path(bag_path).name, "error": f"only {quiet.sum()} quiet samples"}

    A = acc[quiet] / np.linalg.norm(acc[quiet], axis=1, keepdims=True)
    B = up_imu_t[quiet]
    U, _, Vt = np.linalg.svd(B.T @ A)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R_grav = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T      # a_body = acc @ R_grav
    resid_deg = np.degrees(np.arccos(np.clip((A * (B @ R_grav.T)).sum(1), -1, 1)))

    # azimuth about the (now vertical) body-up axis, from gyro vs attitude rate
    pitch, roll = attitude_rad(R)
    g_body = gyr @ R_grav
    kern = np.ones(GYRO_SMOOTH_N) / GYRO_SMOOTH_N
    gx = np.convolve(g_body[:, 0], kern, "same")
    gy = np.convolve(g_body[:, 1], kern, "same")
    tgt = np.column_stack([np.interp(it, ot, np.gradient(pitch, ot)),
                           np.interp(it, ot, np.gradient(roll, ot))])
    W, *_ = np.linalg.lstsq(np.column_stack([gx, gy]), tgt, rcond=None)
    u, sv, vt = np.linalg.svd(W)
    rot = u @ vt
    phi = float(np.arctan2(rot[1, 0], rot[0, 0]))
    pred = np.column_stack([gx, gy]) @ W
    corr = [float(np.corrcoef(pred[:, k], tgt[:, k])[0, 1]) for k in range(2)]

    # Rz(-phi) REMOVES the measured azimuth. Composing R_az.T here would apply
    # Rz(+phi) instead and leave 2*phi of error — that shipped once, see
    # MOUNT_PROVENANCE below.
    c, s = np.cos(-phi), np.sin(-phi)
    R_az = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return {
        "bag": Path(bag_path).name,
        "R": R_grav @ R_az,
        "R_gravity_only": R_grav,
        "n_quiet": int(quiet.sum()),
        "gravity_resid_deg": float(np.median(resid_deg)),
        "azimuth_deg": float(np.degrees(phi)),
        "azimuth_singular_values": [float(v) for v in sv],
        "azimuth_corr": corr,
        # conditioning score: the gyro fit is trustworthy only when it behaves
        # like a rotation AND explains the attitude rates
        "azimuth_quality": float(min(corr) * (1.0 - abs(sv[0] - sv[1]))),
    }


def solve_corpus(bag_paths) -> dict:
    """One mount rotation for the whole corpus (the sensor is rigidly mounted).

    Gravity: quiet samples are POOLED across bags before the Kabsch solve, since
    every bag measures the same rigid rotation and a single 30-40 s run is a small
    sample. Azimuth: taken from the best-conditioned bag rather than averaged — on
    a flat run the azimuth is not a noisy measurement of the truth, it is
    unobserved, so averaging it in adds bias rather than reducing variance.
    """
    A_all, B_all, per_bag = [], [], []
    for bag in bag_paths:
        it, acc, gyr, ot, quat, _ = _read(Path(bag))
        if it.size == 0 or ot.size == 0:
            continue
        R = quat_to_matrix(quat)
        up_body = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, 1.0]))
        up_imu_t = np.stack([np.interp(it, ot, up_body[:, k]) for k in range(3)], -1)
        up_imu_t /= np.linalg.norm(up_imu_t, axis=1, keepdims=True)
        quiet = ((np.abs(np.linalg.norm(acc, axis=1) - GRAVITY) < QUIET_TOL)
                 & (np.linalg.norm(gyr, axis=1) < QUIET_GYRO_RADS))
        if quiet.sum() < MIN_QUIET:
            continue
        A_all.append(acc[quiet] / np.linalg.norm(acc[quiet], axis=1, keepdims=True))
        B_all.append(up_imu_t[quiet])
        per_bag.append(solve_bag(bag))

    if not A_all:
        raise RuntimeError("no bag yielded enough quiet samples")
    A, B = np.concatenate(A_all), np.concatenate(B_all)
    U, _, Vt = np.linalg.svd(B.T @ A)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R_grav = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    resid = np.degrees(np.arccos(np.clip((A * (B @ R_grav.T)).sum(1), -1, 1)))

    ok = [r for r in per_bag if "error" not in r]
    best = max(ok, key=lambda r: r["azimuth_quality"])
    phi = np.radians(best["azimuth_deg"])
    c, s = np.cos(-phi), np.sin(-phi)
    R_az = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return {
        "R": R_grav @ R_az,
        "n_pooled": int(A.shape[0]),
        "gravity_resid_deg": float(np.median(resid)),
        "azimuth_from": best["bag"],
        "azimuth_deg": best["azimuth_deg"],
        "azimuth_quality": best["azimuth_quality"],
        "per_bag": ok,
    }


def verify(bag_path: Path, R_mount: np.ndarray) -> dict:
    """Two independent checks that the fit never saw.

    - `yawrate`: gyro z in the body frame vs the published odometry yaw rate.
      Azimuth-invariant, so this validates the gravity (tilt) part.
    - `gravity_resid`: per-sample angle between the measured acceleration and the
      gravity direction predicted from the INS attitude, on quiet samples.

    Note the mean body acceleration is NOT expected to be `[0, 0, g]`: it is
    `g * mean(up_body)`, so a run with mean pitch p shows `g*sin(p)` on the
    forward axis. That is the vehicle's mean attitude, not a calibration error.
    """
    it, acc, gyr, ot, quat, yawrate = _read(bag_path)
    R = quat_to_matrix(quat)
    up_body = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, 1.0]))
    up_imu_t = np.stack([np.interp(it, ot, up_body[:, k]) for k in range(3)], -1)
    up_imu_t /= np.linalg.norm(up_imu_t, axis=1, keepdims=True)

    a_body = acc @ R_mount
    g_body = gyr @ R_mount
    quiet = ((np.abs(np.linalg.norm(acc, axis=1) - GRAVITY) < QUIET_TOL)
             & (np.linalg.norm(gyr, axis=1) < QUIET_GYRO_RADS))
    meas = a_body[quiet] / np.linalg.norm(a_body[quiet], axis=1, keepdims=True)
    resid = np.degrees(np.arccos(np.clip((meas * up_imu_t[quiet]).sum(1), -1, 1)))

    gz = np.interp(ot, it, g_body[:, 2])
    return {
        "mean_accel_body": [round(float(v), 3) for v in a_body.mean(0)],
        "expected_mean_accel_body": [round(float(v), 3)
                                     for v in GRAVITY * up_imu_t.mean(0)],
        "gravity_resid_deg": float(np.median(resid)),
        "yawrate_corr": float(np.corrcoef(gz, yawrate)[0, 1]),
        "yawrate_scale": float(np.polyfit(gz, yawrate, 1)[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bags", nargs="+", type=Path)
    args = ap.parse_args()

    sol = solve_corpus(args.bags)
    for r in sol["per_bag"]:
        print(f"{r['bag']}: gravity resid {r['gravity_resid_deg']:.2f} deg "
              f"(n_quiet={r['n_quiet']})  azimuth {r['azimuth_deg']:+.2f} deg "
              f"sv={np.round(r['azimuth_singular_values'], 3)} "
              f"corr={np.round(r['azimuth_corr'], 3)} "
              f"quality={r['azimuth_quality']:.3f}")

    print(f"\npooled gravity solve: n={sol['n_pooled']} quiet samples, "
          f"median resid {sol['gravity_resid_deg']:.2f} deg")
    print(f"azimuth from {sol['azimuth_from']} "
          f"({sol['azimuth_deg']:+.2f} deg, quality {sol['azimuth_quality']:.3f})")
    print("\nMOUNT_ROTATION = np.array([")
    for row in sol["R"]:
        print("    [" + ", ".join(f"{v:+.6f}" for v in row) + "],")
    print("], dtype=np.float64)")

    print("\nverification — neither quantity below entered the fit:")
    for bag in args.bags:
        try:
            v = verify(Path(bag), sol["R"])
        except Exception as e:                                # noqa: BLE001
            print(f"  {Path(bag).name}: {type(e).__name__}: {e}")
            continue
        print(f"  {Path(bag).name}:")
        print(f"      mean accel body {v['mean_accel_body']}  vs expected from "
              f"mean attitude {v['expected_mean_accel_body']}")
        print(f"      gravity resid {v['gravity_resid_deg']:.2f} deg   "
              f"yaw-rate corr {v['yawrate_corr']:+.4f} scale {v['yawrate_scale']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
