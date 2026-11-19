# -*- coding: utf-8 -*-
"""Reproduce the FE-PV ML pipeline of the manuscript + LOPO validation.

Outputs: console report of
  Part 1  PCE - RFE feature-count loop (1-10)          -> manuscript Fig. 3b/3c, Table 4 basis
  Part 2  Voc - GBRT, 10 features                      -> manuscript Fig. 5
  Part 3  Jsc - Ridge/DT/RF/GBDT, RFE k=8              -> manuscript Fig. 6, Table 2
  Part 4  LOPO (leave-one-paper-out) cross-validation  -> manuscript Table 4

Random seeds are fixed (split=10, RFE=42, models=7) so results are reproducible.
Tested with Python 3.12, scikit-learn 1.9, pandas 3.0, numpy 2.4.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneGroupOut
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

warnings.filterwarnings("ignore")

data = pd.read_csv("../data/FE_PV_dataset.csv")
data = data.rename(columns={"Irrad": "Irrad "})  # keep original feature name used in the study

# ---- feature engineering (as in the manuscript, Section 2.2) ----
data["WFT"] = data.TopElec.map({"Au": 5.4, "Pt": 5.3, "ITO": 4.8, "Ag": 4.2, "Al": 4.2, "Cu": 4.6})
data["WFB"] = data.BottomElec.map({"Au": 5.4, "Pt": 5.3, "FTO": 4.4, "ITO": 4.8, "SRO": 5.2, "LSMO": 4.96, "SNTO": 4.1})
data["WFDiff"] = data["WFB"] - data["WFT"]    # dPhi
data["WBBDiff"] = data["WFB"] - data["Bandgap"]  # dBE
data["WTBDiff"] = data["WFT"] - data["Bandgap"]  # dTE

FEATS = ["Bandgap", "Pr", "PoS", "Thick", "Irrad ", "WFT", "WFB", "WFDiff", "WBBDiff", "WTBDiff"]

GBDT_GRID = {"n_estimators": [10, 15, 20, 25, 30],
             "max_depth": [3, 4, 5, 6, 8, 10, None],
             "learning_rate": [0.01, 0.1, 1]}


def prep(target):
    X = data[FEATS].copy()
    y = data[target].copy()
    return train_test_split(X, y, test_size=0.20, random_state=10)


def scale(X_tr, X_te):
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te)


def eval_gs(gs, X_tr_s, X_te_s, y_tr, y_te):
    y_pred = gs.predict(X_te_s)
    return dict(train_R2=gs.score(X_tr_s, y_tr), test_R2=gs.score(X_te_s, y_te),
                MAE=metrics.mean_absolute_error(y_te, y_pred),
                MSE=metrics.mean_squared_error(y_te, y_pred), best=gs.best_params_)


print("=" * 78)
print("PART 1  PCE - RFE feature-count loop (paper Fig.3b/3c)")
print("=" * 78)
X_tr, X_te, y_tr, y_te = prep("PCE")
for k in range(1, 11):
    rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=k)
    X_tr_r = rfe.fit_transform(X_tr, y_tr)
    X_te_r = rfe.transform(X_te)
    X_tr_s, X_te_s = scale(X_tr_r, X_te_r)
    gs = GridSearchCV(GradientBoostingRegressor(random_state=7), GBDT_GRID, scoring="r2", cv=10, n_jobs=-1)
    gs.fit(X_tr_s, y_tr)
    r = eval_gs(gs, X_tr_s, X_te_s, y_tr, y_te)
    sel = [FEATS[i] for i in np.where(rfe.get_support())[0]]
    print(f"k={k:2d}  trainR2={r['train_R2']:.3f}  testR2={r['test_R2']:.3f}  "
          f"MAE={r['MAE']:.4f}  MSE={r['MSE']:.5f}  sel={sel}")

print()
print("=" * 78)
print("PART 2  Voc - GBRT, 10 features (paper Fig. 5)")
print("=" * 78)
X_tr, X_te, y_tr, y_te = prep("Voc")
X_tr_s, X_te_s = scale(X_tr, X_te)
gs = GridSearchCV(GradientBoostingRegressor(random_state=7), GBDT_GRID, scoring="r2", cv=10, n_jobs=-1)
gs.fit(X_tr_s, y_tr)
print(eval_gs(gs, X_tr_s, X_te_s, y_tr, y_te))

print()
print("=" * 78)
print("PART 3  Jsc - four models, RFE k=8 (paper Fig. 6, Table 2)")
print("=" * 78)
X_tr, X_te, y_tr, y_te = prep("JscmA")
rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=8)
X_tr_r = rfe.fit_transform(X_tr, y_tr)
X_te_r = rfe.transform(X_te)
print("selected:", [FEATS[i] for i in np.where(rfe.get_support())[0]])
X_tr_s, X_te_s = scale(X_tr_r, X_te_r)
grids = {
    "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1.0, 10]}),
    "DT": (DecisionTreeRegressor(random_state=7), {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]}),
    "RF": (RandomForestRegressor(random_state=7), {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]}),
    "GBDT": (GradientBoostingRegressor(random_state=7), GBDT_GRID),
}
for name, (est, grid) in grids.items():
    gs = GridSearchCV(est, grid, scoring="r2", cv=10, n_jobs=-1)
    gs.fit(X_tr_s, y_tr)
    print(name, eval_gs(gs, X_tr_s, X_te_s, y_tr, y_te))

print()
print("=" * 78)
print("PART 4  LOPO cross-validation (paper Table 4, Section 3.4)")
print("=" * 78)
groups = data["Author"].astype(str) + "_" + data["Year"].astype(str)
print("papers:", sorted(groups.unique()), " n =", groups.nunique())
logo = LeaveOneGroupOut()
for target, label, mk in [
    ("PCE", "PCE", lambda: GradientBoostingRegressor(random_state=7, n_estimators=15, max_depth=4, learning_rate=0.1)),
    ("Voc", "Voc", lambda: GradientBoostingRegressor(random_state=7, n_estimators=30, max_depth=8, learning_rate=0.1)),
    ("JscmA", "Jsc", lambda: DecisionTreeRegressor(random_state=7, max_depth=7)),
]:
    X_all, y_all = data[FEATS].copy(), data[target].copy()
    y_true, y_pred = [], []
    rows = []
    for tr_i, te_i in logo.split(X_all, y_all, groups):
        sc = StandardScaler().fit(X_all.iloc[tr_i])
        m = mk()
        m.fit(sc.transform(X_all.iloc[tr_i]), y_all.iloc[tr_i])
        pred = m.predict(sc.transform(X_all.iloc[te_i]))
        y_true += list(y_all.iloc[te_i]); y_pred += list(pred)
        rows.append((groups.iloc[te_i].iloc[0], len(te_i), metrics.r2_score(y_all.iloc[te_i], pred)))
    print(f"\n[{label}] LOPO pooled R2 = {metrics.r2_score(y_true, y_pred):.3f}   "
          f"MAE = {metrics.mean_absolute_error(y_true, y_pred):.4f}")
    for g, n, r2g in rows:
        print(f"   {g:16s} n={n:2d}  R2={r2g:10.2f}")
