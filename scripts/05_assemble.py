import os
import pandas as pd
import joblib
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, "..", "data")
results_dir = os.path.join(root, "..", "results")
assets_dir = os.path.join(root, "..", "assets")

de = pd.read_csv(os.path.join(data_dir, "enamine_stock_annotations.csv"))
df = pd.read_csv(os.path.join(results_dir, "promiscuity_pxf_predictions_enamine_stock.csv"))
df = df.drop(columns="smiles")
de = de.merge(df, on="catalog_id", how="left")
df = pd.read_csv(os.path.join(results_dir, "promiscuity_pxf_fxp_predictions_enamine_stock.csv"))
df = df.drop(columns="smiles")
de = de.merge(df, on="catalog_id", how="left")
df = pd.read_csv(os.path.join(results_dir, "signature_predictions_enamine_stock.csv"))
df = df.drop(columns="smiles")
de = de.merge(df, on="catalog_id", how="left")

de.to_csv(os.path.join(assets_dir, "enamine_stock_predictions_enamine_stock.csv"), index=False)

tasks = [c for c in list(de.columns) if c.startswith("promiscuity_") or c.startswith("signature_")]

R = []
for t in tasks:
    mdl = joblib.load(os.path.join(results_dir, t+".joblib"))
    print(mdl.validation_metrics)
    aucs = mdl.validation_metrics["aucs"]
    r = [np.mean(aucs), np.std(aucs), np.min(aucs), np.max(aucs)]
    R += [[t] + r]

df = pd.DataFrame(R, columns=["task", "auroc_mean", "auroc_std", "auroc_min", "auroc_max"])
df.to_csv(os.path.join(assets_dir, "crossvalidation_results.csv"), index=False)