# Vendored robot description assets

`anymal_c.urdf` — ANYbotics `anymal_c_simple_description` (BSD-3-Clause), fetched
from
https://raw.githubusercontent.com/ANYbotics/anymal_c_simple_description/master/urdf/anymal.urdf

Used by `tools/legged/urdf_fk.py` for base-frame foot FK in the Isaac-legged
stage-2 `foot_slip` label. Only the joint tree (origins/axes) is consumed — mesh references are never resolved. The IsaacLab
ANYmal-C USD derives from this same description; the extractor's per-episode
FK-vs-GT swing check (`extract_summary.json`) guards against any geometry drift.
