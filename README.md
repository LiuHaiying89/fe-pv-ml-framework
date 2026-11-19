# Machine Learning Framework for Ferroelectric Perovskite Oxide Photovoltaics

Code and data for the manuscript:

> *A machine learning framework for accelerated power conversion efficiency of ferroelectric perovskite oxides photovoltaic device* (submitted to *Computational Materials Science*).

## Overview

This repository provides the complete, reproducible pipeline for building machine-learning
prediction models of the power conversion efficiency (PCE), open-circuit voltage (Voc) and
short-circuit current density (Jsc) of electrode/ferroelectric-film/electrode (E/FE/E) photovoltaic
devices based on ferroelectric perovskite oxides. It includes:

- the curated experimental dataset (50 devices from 9 literature sources),
- the physics-informed feature-engineering and model-training code,
- leave-one-paper-out (LOPO) transferability validation,
- learning-curve analysis of dataset-size dependence.

## Repository structure

```
fe-pv-ml-framework/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   └── FE_PV_dataset.csv     # 50 experimental data points (9 literature sources)
├── code/
│   ├── 01_reproduce_models.py  # feature engineering, RFE loop, model training, LOPO validation
│   └── 02_learning_curve.py    # learning curves (test R^2 vs training-set size)
└── figures/                    # generated outputs (learning_curve.png, learning_curve_results.csv)
```

## Dataset

`data/FE_PV_dataset.csv` contains 50 experimental records compiled from published studies of
E/FE/E ferroelectric photovoltaic devices. Raw columns: author, year, material, band gap (eV),
remanent polarization Pr (μC/cm²), polarization state PoS (−1/0/1), top/bottom electrode,
film thickness (μm), wavelength (nm), light intensity (mW/cm²), Voc (V), Jsc (μA/cm²),
fill factor, PCE (%), JscmA (mA/cm²). The ten model features (Section 2.2 of the manuscript)
are engineered in the scripts from these raw values.

## Reproduction

```bash
pip install -r requirements.txt
cd code
python 01_reproduce_models.py   # -> console report; LOPO numbers match manuscript Table 4
python 02_learning_curve.py     # -> ../figures/learning_curve.png (manuscript Fig. 8)
```

Tested with Python 3.12, scikit-learn 1.9, pandas 3.0, numpy 2.4.

All random seeds are fixed (train/test split = 10, RFE = 42, model = 7), so results are
deterministic and reproduce the manuscript exactly.

## Mapping between code and manuscript

| Script output                              | Manuscript element                |
|--------------------------------------------|-----------------------------------|
| Part 1 (PCE RFE loop, k = 1–10)            | Fig. 3b–3d (feature-selection curve, optimal 6 features) |
| Part 2 (Voc, GBRT, 10 features)            | Fig. 5                            |
| Part 3 (Jsc, four models, RFE k = 8)       | Fig. 6 and Table 2                |
| Part 4 (LOPO cross-validation)             | Section 3.4 and Table 4           |
| `02_learning_curve.py`                   | Fig. 8                            |

## License

- Code: MIT License (see LICENSE).
- Data: CC BY 4.0. The dataset values are compiled from the published literature; please cite
  the original experimental papers (listed in the manuscript references) alongside this repository.

## Citation

If you use this repository, please cite the manuscript and the dataset DOI:
https://doi.org/10.5281/zenodo.24357817 (release v1.0.0, archived on Zenodo).
