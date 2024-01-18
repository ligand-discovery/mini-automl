import os
import pandas as pd
import joblib
import numpy as np
from fragmentembedding import FragmentEmbedder
from fragmentembedding.physchem_desc import PhyschemDescriptor
from miniautoml.binary_classifier_2 import train_binary_classifier

root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, "..", "data")
results_dir = os.path.join(root, "..", "results")

df = pd.read_csv(os.path.join(data_dir, "fragment_signatures.csv"))

smiles_list = df["smiles"].tolist()

descriptor = PhyschemDescriptor(discretize=False)
descriptor.fit(smiles_list)
X = descriptor.transform(smiles_list)

y_columns = list(df.columns[2:])
for yc in y_columns:
    y = df[yc].tolist()
    mdl = train_binary_classifier(X, y, n_splits=10)
    aurocs = mdl.validation_metrics["aucs"]
    print(yc, np.mean(aurocs), aurocs)
    joblib.dump(mdl, os.path.join(results_dir, yc+"_2.joblib"))