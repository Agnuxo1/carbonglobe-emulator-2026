# Third-party components and data

## Data (not redistributed)
- **CarbonGlobe** (Wang et al., NeurIPS 2025 Datasets & Benchmarks). Present-day
  global benchmark used for training. Kaggle dataset `zhihaow/carbonglobe`;
  code for building it at https://github.com/zhwang0/carbon-globe.
- **IEEE BigData Cup 2026 — AI Emulation Challenge** test data (Kaggle competition
  `ieee-bigdata-cup-2026-ai-emulation-challenge`). The competition rules license the
  data for "Use, Non-Commercial and Academic Research Only", so no part of either
  dataset is included in this repository. The code downloads what it needs with the
  official Kaggle API.
- The target variables are produced by the **Ecosystem Demography (ED)** process-based
  model, which this work emulates; it is not redistributed either.

The data layout and the December-of-year-40 target convention were read from the
organizers' own tutorial notebook `01_data_tutorial.ipynb`, distributed from the
Google Drive folder linked on the competition overview page.

## Software
- **holographic-reservoir** (Apache-2.0), https://pypi.org/project/holographic-reservoir/ —
  Francisco Angulo de Lafuente's own reservoir-computing library. The reservoir weights
  used here are produced by its own `HolographicReservoirModel.__post_init__`; this
  repository only batches its identical state-update rule over sequences on the GPU, a
  port verified to 8e-8 in `tests/test_hrc_port.py`.
- **pixelflow / pixelflow-rc** (Apache-2.0) — the same author's GPU texture reservoir.
  Considered for spatial coupling and **discarded with a measured reason**: the 6171
  competition sites cannot be located on the 0.5 degree grid (see README), so no spatial
  neighbourhood can be built for them.
- NumPy, pandas, SciPy, PyTorch, LightGBM, the Kaggle CLI — all under their own
  OSI-approved licences.

## Citation of the benchmark
```
@inproceedings{wang2025carbonglobe,
  title  = {CarbonGlobe: A Global-Scale, Multi-Decade Dataset and Benchmark for
            Carbon Forecasting in Forest Ecosystems},
  author = {Wang, Zhihao and Ma, Lei and Hurtt, George and Jia, Xiaowei and Li, Yanhua
            and Li, Rui and Li, Zhili and Xu, Shaowen and Xie, Yiqun},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume = {38}, year = {2025}
}
```
