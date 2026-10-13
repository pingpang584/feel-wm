# Feel-WM

Code for **Feeling Terrain Before Crossing: World Models for Off-Road Navigation**
([arXiv:2609.19863](https://arxiv.org/abs/2609.19863)).

## Install

```bash
conda create -n feelwm python=3.10 && conda activate feelwm
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121   # or your CUDA build
pip install -e ".[preprocess,features,test]"      # add ,legged for GrandTour
pytest -q
```

## Data

| Dataset | Source |
|---|---|
| TartanDrive 2.0 | [castacks/tartan_drive_2.0](https://github.com/castacks/tartan_drive_2.0) |
| TartanDrive 1.0 | [castacks/tartan_drive](https://github.com/castacks/tartan_drive) |
| TartanGround | [tartanair.org/tartanground](https://tartanair.org/tartanground/) (download via [tartanairpy](https://github.com/castacks/tartanairpy)) |
| GrandTour | [leggedrobotics/grand_tour_dataset](https://github.com/leggedrobotics/grand_tour_dataset) |

Set the data and output roots (defaults `./datasets`, `./runs`):

```bash
export DATA_ROOT=/path/to/datasets OUTPUT_ROOT=/path/to/runs
```

**TartanDrive 2.0** — bags in `${DATA_ROOT}/tartandrive/<seq>/`, then:

```bash
python tools/tartandrive/build_all.py                          # labels + proprio windows
python tools/tartandrive/derive_binary_safety.py               # failure-risk labels
python tools/features/cache_dino_features_reg.py --gpus 0,1,2,3
python tools/features/consolidate_dino_features.py --feat-version dinov2reg_v1
```

**TartanDrive 1.0** — `tools/tartandrive/td1_fetch_build_loop.sh` (set `TD1_ENDPOINT` to the
archive URL from the official download instructions).

**TartanGround + GrandTour** — `tools/legged/extract_tartanground.py`,
`tools/legged/extract_grandtour.py`, then `tools/legged/cache_dino_features.py`.

Splits and normalization statistics are in `data/`.

## Configs

One file per domain; `--variant` selects a model.

| File | Variants |
|---|---|
| `configs/tartandrive.yaml` | `td2` (default), `td2_noprop`, `td1`, `td1_td2eval` |
| `configs/gazebo.yaml` | `feelwm` (default), `noprop`, `visual_only` |
| `configs/legged_isaac.yaml` | `feelwm` (default), `noprop`, `visual_only` |
| `configs/legged_tg_gt.yaml` | — |
| `configs/planning.yaml` | `wheeled` (default), `legged_tg_gt`, `legged_isaac` |

`noprop` = Feel-WM without proprioceptive input; `visual_only` = no proprioceptive input or
heads.

## Training

```bash
torchrun --nproc_per_node=8 tools/train.py configs/tartandrive.yaml --variant td2
python tools/train.py configs/legged_tg_gt.yaml --legged
```

## Evaluation

Proprioceptive-feature R² and failure-risk AUC:

```bash
python tools/eval_offline.py configs/tartandrive.yaml \
    ${OUTPUT_ROOT}/tartandrive/td2/checkpoint_best.pt --variant td2 --split val
```

Open-loop CEM planning tools are in `tools/eval/` (usage in each file's docstring).

## License

MIT. The vendored ANYmal-C URDF is BSD-3-Clause (ANYbotics); see `NOTICE`.

## Citation

```bibtex
@article{son2026feeling,
  title         = {Feeling Terrain Before Crossing: World Models for Off-Road Navigation},
  author        = {Son, E-In and Kim, Dong-Wook and Hwang, Ji-Hoon and Lee, Kangsun and
                   Bae, Jisung and Kim, Jung-Taak and Seo, Seung-Woo},
  journal       = {arXiv preprint arXiv:2609.19863},
  year          = {2026},
  eprint        = {2609.19863},
  archivePrefix = {arXiv},
}
```
