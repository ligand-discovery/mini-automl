import os
import pandas as pd
import numpy as np
import joblib
from fragmentembedding import FragmentEmbedder

root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, "..", "data")
results_dir = os.path.join(root, "..", "results")

df = pd.read_csv(os.path.join(data_dir, "enamine_stock.csv"))
ds = pd.read_csv(os.path.join(data_dir, "fragment_signatures.csv"))
headers = list(ds.columns)[2:]

smiles_list = df["smiles"].tolist()
identifiers = df["catalog_id"].tolist()
X = FragmentEmbedder().transform(smiles_list)

R = []
for h in headers:
    print(h)
    mdl = joblib.load(os.path.join(results_dir, h+".joblib"))
    y_hat = list(mdl.predict(X))
    R += [y_hat]
Y_hat = np.array(R).T

R = []
for i in range(Y_hat.shape[0]):
    r = [identifiers[i], smiles_list[i]] + [y_ for y_ in Y_hat[i,:]]
    R += [r]

df = pd.DataFrame(R, columns = ["catalog_id", "smiles"] + headers)
df.to_csv(os.path.join(results_dir, "signature_predictions_enamine_stock.csv"), index=False)