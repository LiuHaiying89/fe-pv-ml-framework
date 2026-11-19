# -*- coding: utf-8 -*-
"""Learning-curve analysis: test R^2 vs training-set size for the three optimal models.
Fixed 80/20 split (random_state=10, as in the paper). Features selected once on the full
training set (RFE k=6 for PCE, all 10 for Voc, RFE k=8 for Jsc). Each size repeated over
20 subsampling seeds; report mean +/- std of test R^2.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

warnings.filterwarnings("ignore")

data = pd.read_csv("../data/FE_PV_dataset.csv")
data = data.rename(columns={"Irrad": "Irrad "})
data["WFT"] = data.TopElec.map({"Au": 5.4, "Pt": 5.3, "ITO": 4.8, "Ag": 4.2, "Al": 4.2, "Cu": 4.6})
data["WFB"] = data.BottomElec.map({"Au": 5.4, "Pt": 5.3, "FTO": 4.4, "ITO": 4.8, "SRO": 5.2, "LSMO": 4.96, "SNTO": 4.1})
data["WFDiff"] = data["WFB"] - data["WFT"]
data["WBBDiff"] = data["WFB"] - data["Bandgap"]
data["WTBDiff"] = data["WFT"] - data["Bandgap"]
FEATS = ["Bandgap", "Pr", "PoS", "Thick", "Irrad ", "WFT", "WFB", "WFDiff", "WBBDiff", "WTBDiff"]

FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
N_SEEDS = 20

TARGETS = {
    "PCE": dict(col="PCE",    k=6,  model=lambda: GradientBoostingRegressor(random_state=7, n_estimators=15, max_depth=4, learning_rate=0.1)),
    "Voc": dict(col="Voc",    k=None, model=lambda: GradientBoostingRegressor(random_state=7, n_estimators=30, max_depth=None, learning_rate=1)),
    "Jsc": dict(col="JscmA",  k=8,  model=lambda: DecisionTreeRegressor(random_state=7, max_depth=7)),
}

results = []
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
for ax, (name, cfg) in zip(axes, TARGETS.items()):
    X = data[FEATS].copy()
    y = data[cfg["col"]].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=10)
    if cfg["k"] is not None:
        rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=cfg["k"])
        X_tr = pd.DataFrame(rfe.fit_transform(X_tr, y_tr), columns=[FEATS[i] for i in np.where(rfe.get_support())[0]], index=y_tr.index)
        X_te = pd.DataFrame(rfe.transform(X_te), columns=X_tr.columns, index=y_te.index)
    sc = StandardScaler().fit(X_tr)
    X_te_s = sc.transform(X_te)
    means, stds = [], []
    for f in FRACS:
        r2s = []
        n_sub = max(4, int(round(len(X_tr) * f)))
        for s in range(N_SEEDS):
            rng = np.random.RandomState(1000 + s)
            idx = rng.choice(len(X_tr), size=n_sub, replace=False)
            m = cfg["model"]()
            m.fit(sc.transform(X_tr.iloc[idx]), y_tr.iloc[idx])
            r2s.append(metrics.r2_score(y_te, m.predict(X_te_s)))
        means.append(np.mean(r2s)); stds.append(np.std(r2s))
        print(f"{name}: train_size={n_sub:2d} ({f:.0%})  testR2 = {means[-1]:.3f} +/- {stds[-1]:.3f}")
    for i, f in enumerate(FRACS):
        results.append([name, f, int(round(len(X_tr) * f)), means[i], stds[i]])
    xs = [int(round(len(X_tr) * f)) for f in FRACS]
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=4, color="#7B9EF8", ecolor="#B9C8F5")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(name, fontsize=14)
    ax.set_xlabel("Training set size", fontsize=12)
    ax.set_xticks(xs)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Test $R^2$", fontsize=12)
fig.tight_layout()
fig.savefig("../figures/learning_curve.png", dpi=300, bbox_inches="tight")

pd.DataFrame(results, columns=["target", "fraction", "train_size", "test_R2_mean", "test_R2_std"]).to_csv(
    "../figures/learning_curve_results.csv", index=False)
print("\nsaved: learning_curve.png, learning_curve_results.csv")
