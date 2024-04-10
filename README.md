# Mini-AutoML: Simple AutoML functionalities for the Ligand Discovery project
This repository contains a simple AutoML tool based on [TabPFN](https://github.com/automl/TabPFN) to quickly perform binary classification tasks.

## Installation

Please follow these steps to install `mini-automl`:

```bash
# create a conda environment
conda create -n miniautoml python=3.10
conda activate miniautoml

# download tabpfn for cpu usage
pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/DhanshreeA/TabPFN.git
pip install TabPFN/.
rm -rf TabPFN

# download fragment embeddings
git clone https://github.com/ligand-discovery/fragment-embedding.git
pip install -e fragment-embedding/.

# clone the current repository
git clone https://github.com/ligand-discovery/mini-automl
cd mini-automl
python -m pip install -e .
```

## Usage

```python
from fragmentembedding import FragmentEmbedder
from miniautoml import train_binary_classifier, get_example

df = get_example()
smiles_list = df["smiles"].tolist()
y = df["signature_2"].tolist()

X = FragmentEmbedder().transform(smiles_list)
mdl = train_binary_classifier(X[:-10], y[:-10], n_splits=5)

mdl.predict(X[-10:])
```

The code above will automatically perform a stratified shuffle splits to estimate model performance. If you just want to train the model, simply set `n_splits=None`.

# Data precalculations

To pretrain promiscuity and signature models, execute the following:

```bash
python scripts/00_data_preparation.py
python scripts/01_train_promiscuity.py
python scripts/02_predict_promiscuity.py
python scripts/03_train_signatures.py
python scripts/04_predict_signatures.py
python scripts/05_assemble.py
```

## About

This work is done by the [Georg Winter Lab](https://www.winter-lab.com/) at [CeMM](https://cemm.at), Vienna.