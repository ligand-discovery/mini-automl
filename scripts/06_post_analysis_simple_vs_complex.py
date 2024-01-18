import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
import numpy as np
import collections
from mordred import Calculator, descriptors
from rdkit import Chem
from fragmentembedding import FragmentEmbedder
from tqdm import tqdm
from miniautoml.binary_classifier import train_binary_classifier

root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, "..", "data")
results_dir = os.path.join(root, "..", "results")
assets_dir = os.path.join(root, "..", "assets")

df = pd.read_csv(os.path.join(data_dir, "promiscuity_pxf.csv"))

smiles_list = df["smiles"].tolist()

X = FragmentEmbedder().transform(smiles_list)

calc = Calculator(descriptors, ignore_3D=True)

def csp2_calculate(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    csp2_count = sum(atom.GetHybridization() == Chem.HybridizationType.SP2 for atom in molecule.GetAtoms() if atom.GetSymbol() == 'C')
    return csp2_count

logp = list(np.array([MolLogP(Chem.MolFromSmiles(smi)) for smi in smiles_list]))
csp2 = list(np.array([csp2_calculate(smi) for smi in smiles_list]))

def loo_predictions(X, y):
    idxs = np.array([i for i in range(X.shape[0])], dtype=int)
    y_hat = []
    for idx in tqdm(idxs):
        train_idxs = [i for i in idxs if i != idx]
        test_idxs = [idx]
        mdl = train_binary_classifier(X[train_idxs], y[train_idxs], n_splits=None)
        y_hat += list(mdl.predict(X[test_idxs]))
    return y_hat

y_columns = list(df.columns[2:])
for yc in y_columns:
    file_name = os.path.join(results_dir, "complex_vs_simple", "{0}.csv".format(yc))
    if os.path.exists(file_name):
        continue
    y = np.array(df[yc].tolist())
    y_hat = loo_predictions(X, y)
    data_ = collections.OrderedDict()
    data_["fid"] = df["fid"]
    data_["smiles"] = df["smiles"]
    data_["logp"] = logp
    data_["csp2"] = csp2
    data_["y"] = y
    data_["y_hat"] = y_hat
    pd.DataFrame(data_).to_csv(file_name, index=False)

df = pd.read_csv(os.path.join(data_dir, "promiscuity_pxf_fxp.csv"))

y_columns = list(df.columns[2:])
for yc in y_columns:
    file_name = os.path.join(results_dir, "complex_vs_simple", "{0}.csv".format(yc))
    if os.path.exists(file_name):
        continue
    y = np.array(df[yc].tolist())
    y_hat = loo_predictions(X, y)
    data_ = collections.OrderedDict()
    data_["fid"] = df["fid"]
    data_["smiles"] = df["smiles"]
    data_["logp"] = logp
    data_["csp2"] = csp2
    data_["y"] = y
    data_["y_hat"] = y_hat
    pd.DataFrame(data_).to_csv(file_name, index=False)
