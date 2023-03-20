import numpy as np
from lol import LOL
from tabpfn import TabPFNClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc


class BinaryClassifier(object):
    def __init__(self, reducer, model, validation_metrics):
        self.reducer = reducer
        self.model = model
        self.validation_metrics = validation_metrics

    def predict(self, X):
        X = self.reducer.transform(X)
        return self.model.predict_proba(X)[:, 1]


def train_binary_classifier(X, y, n_splits=10, test_size=0.2):
    aucs = []
    fprs = []
    tprs = []
    red = LOL(n_components=100)
    mdl = TabPFNClassifier(N_ensemble_configurations=32)
    mdl.remove_models_from_memory()
    if n_splits is not None:
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        for i, (train_index, test_index) in enumerate(splitter.split(X, y)):
            X_train, X_test, y_train, y_test = (
                X[train_index],
                X[test_index],
                y[train_index],
                y[test_index],
            )
            red.fit(X_train, y_train)
            X_train = red.transform(X_train)
            X_test = red.transform(X_test)
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict_proba(X_test)[:, 1]
            mdl.remove_models_from_memory()
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            tprs += [tpr]
            fprs += [fpr]
            aucs += [auc(fpr, tpr)]
        vd = {
            "aucs": aucs,
            "positives": int(np.sum(y)),
            "fprs": fprs,
            "tprs": tprs,
            "n_splits": n_splits,
            "test_size": test_size,
        }
    else:
        vd = None
    red.fit(X, y)
    X = red.transform(X)
    mdl.fit(X, y)
    return BinaryClassifier(red, mdl, vd)
