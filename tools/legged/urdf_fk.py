"""Minimal URDF forward kinematics for the ANYmal foot chains.

Dependency-light on purpose (numpy + scipy + stdlib xml): no pinocchio or other
rigid-body library is needed to extract Isaac legged bags. Geometry is never hand-typed — the
joint tree (origins/axes) is parsed from the vendored URDF (`assets/anymal_c.urdf`,
see assets/README.md), so the only trust anchor is that file.

Exposes vectorized base-frame foot positions for the 12 canonical joint angles
(canonical order: LF_HAA LF_HFE LF_KFE | RF | LH | RH). The stage-2
foot_slip label differentiates these positions over time (dp/dt == J·qd by the
chain rule), so no explicit Jacobian is needed.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CANONICAL_JOINTS = [f"{leg}_{j}" for leg in ("LF", "RF", "LH", "RH")
                    for j in ("HAA", "HFE", "KFE")]
FOOT_LINKS = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
DEFAULT_URDF = Path(__file__).resolve().parent / "assets" / "anymal_c.urdf"


def _origin_mat(xyz, rpy):
    # URDF origin rpy = fixed-axis (extrinsic) x-y-z
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def _rotvec_mats(axis, q):
    """(M,) angles about a fixed local axis -> (M,4,4)."""
    R = Rotation.from_rotvec(np.outer(q, axis)).as_matrix()
    T = np.tile(np.eye(4), (len(q), 1, 1))
    T[:, :3, :3] = R
    return T


class FootFK:
    """Base-frame FK for the four foot links of a leg-serial URDF.

    `foot_coms` holds each FOOT link's inertial origin (foot frame). The PhysX GT
    foot-vel topic reports the link CoM velocity (measured on a pilot episode:
    FK-at-CoM vs GT swing corr 0.991 / ratio 0.997, vs 0.882 / 1.206 at the ball
    center), so FK-vs-GT comparisons must be evaluated at the CoM point, while the
    slip LABEL stays at the ball center (contact-point proxy — correct zero)."""

    def __init__(self, urdf_path: Path = DEFAULT_URDF):
        root = ET.parse(urdf_path).getroot()
        self.foot_coms = np.zeros((4, 3))
        for link in root.iter("link"):
            if link.get("name") in FOOT_LINKS:
                io = link.find("inertial/origin")
                if io is not None:
                    self.foot_coms[FOOT_LINKS.index(link.get("name"))] = \
                        [float(v) for v in io.get("xyz", "0 0 0").split()]
        by_child = {}
        for j in root.iter("joint"):
            o = j.find("origin")
            ax = j.find("axis")
            by_child[j.find("child").get("link")] = {
                "name": j.get("name"), "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "xyz": [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()],
                "rpy": [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()],
                "axis": [float(v) for v in (ax.get("xyz", "1 0 0") if ax is not None else "1 0 0").split()],
            }
        self.chains = []
        for foot in FOOT_LINKS:
            chain, cur = [], foot
            while cur in by_child:
                chain.append(by_child[cur])
                cur = by_child[cur]["parent"]
            if cur != "base":
                raise ValueError(f"{foot}: chain root is {cur!r}, expected 'base'")
            elems = []
            for j in reversed(chain):
                T0 = _origin_mat(j["xyz"], j["rpy"])
                if j["type"] in ("revolute", "continuous"):
                    if j["name"] not in CANONICAL_JOINTS:
                        raise ValueError(f"unexpected actuated joint {j['name']}")
                    elems.append((T0, CANONICAL_JOINTS.index(j["name"]),
                                  np.asarray(j["axis"], float)))
                else:
                    if elems and elems[-1][1] is None:      # fold consecutive fixed
                        elems[-1] = (elems[-1][0] @ T0, None, None)
                    else:
                        elems.append((T0, None, None))
            self.chains.append(elems)

    def foot_transforms(self, q: np.ndarray) -> np.ndarray:
        """q (M,12) canonical order -> (M,4,4,4) base-frame foot transforms."""
        q = np.atleast_2d(np.asarray(q, float))
        M = len(q)
        out = np.empty((M, 4, 4, 4))
        for f, elems in enumerate(self.chains):
            T = np.tile(np.eye(4), (M, 1, 1))
            for T0, qidx, axis in elems:
                T = T @ T0
                if qidx is not None:
                    T = np.einsum("mij,mjk->mik", T, _rotvec_mats(axis, q[:, qidx]))
            out[:, f] = T
        return out

    def foot_positions(self, q: np.ndarray, offsets: np.ndarray = None) -> np.ndarray:
        """q (M,12) -> (M,4,3) base-frame positions of a per-foot point.

        offsets (4,3) in the foot frame; None = the foot frame origin (ball center)."""
        T = self.foot_transforms(q)
        p = T[:, :, :3, 3]
        if offsets is not None:
            p = p + np.einsum("mfij,fj->mfi", T[:, :, :3, :3], np.asarray(offsets, float))
        return p
