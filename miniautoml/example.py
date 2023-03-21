import os
import pandas as pd

root = os.path.abspath(os.path.dirname(__file__))

def get_example():
    fn = os.path.join(root, "..", "data", "fragment_signatures.csv")
    df = pd.read_csv(fn)
    return df[["smiles", "signature_2"]]
