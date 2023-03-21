# Mini-AutoML: Simple AutoML functionalities for the Ligand Discovery project
This repository contains a simple AutoML tool based on [TabPFN](https://github.com/automl/TabPFN) to quickly perform binary classification tasks.

## Installation

The Mini-AutoML module is `pip`-installable:

```bash
git clone https://github.com/ligand-discovery/mini-automl
cd mini-automl
python -m pip install -e .
```

To follow the example below, please also install the [Fragment Embedding](https://github.com/ligand-discovery/fragment-embedding) tool.

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

## About

This work is done by the [Georg Winter Lab](https://www.winter-lab.com/) at [CeMM](https://cemm.at), Vienna.