# FNO-Regenerate

A from-scratch PyTorch reimplementation of four **neural operators** from
*Kovachki, Li et al., "Neural Operator: Learning Maps Between Function Spaces"*,
tested on 2D Navier-Stokes fluid flow (paper Section 7.3).

A neural operator is a neural network that learns the map from one *function* to
another (here: vorticity at time 0 -> vorticity at a later time), so it can stand in for
a slow numerical PDE solver.

| Model | Idea | Paper section |
|---|---|---|
| **FNO** - Fourier Neural Operator | Multiply in Fourier space (FFT -> learned weights -> inverse FFT) | 4.4 |
| **GNO** - Graph Neural Operator | Message passing on a radius graph with a learned kernel | 4.1 |
| **LNO** - Low-rank Neural Operator | Kernel factorised as a sum of R products of basis functions | 4.2 |
| **MGNO** - Multipole Graph Neural Operator | Local + coarse-grid long-range message passing | 4.3 |

## Repository layout

```
neural_operators/      reusable library (import this)
  models/                fno.py, gno.py, lno.py, mgno.py
  layers/                spectral_conv.py, kernel_nn.py
  data/                  grf.py (random initial fields), navier_stokes_2d.py (solver)
  utils.py               relative-L2 loss, training loop, timing helpers
notebooks/             experiments - run in numeric order
  01_generate_navier_stokes_data.ipynb
  02_fno.ipynb ... 05_mgno.ipynb
  06_compare_results.ipynb
  source/                one walkthrough notebook per library module
results/               saved summaries (*.json), weights (*.pt), rollout GIF
requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
jupyter notebook                                     # then open notebooks/
```

Run the notebooks from the `notebooks/` folder. Notebook 01 writes the dataset to
`data_cache/` (not tracked by git); notebooks 02-05 read it.

## Results (64x64 Navier-Stokes, test relative L2 error)

| Model | Parameters | Test rel. L2 | Inference time (s) |
|---|---|---|---|
| FNO  | 1,188,353 | 0.0233 | 0.0056 |
| GNO  |   176,001 | 0.0813 | 0.0308 |
| LNO  |   309,761 | 0.0717 | 0.0100 |
| MGNO |   158,081 | 0.0563 | 0.1289 |

For comparison, the numerical solver takes about 0.096 s for the same problem.
Numbers come from `results/*_summary.json`.

## Git workflow

Work on a feature branch, commit small and often, then push:

```bash
git status
git add <files>
git commit -m "Describe what and why"
git push -u origin <branch>
```
