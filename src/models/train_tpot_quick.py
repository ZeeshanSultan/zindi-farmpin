import pandas as pd
from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

seed = 2019


def train_model(X, y):
    tpot = TPOTClassifier(
        generations=10,
        population_size=20,
        verbosity=2,
        n_jobs=-1,
        scoring='accuracy',
        cv=5
    )

    tpot.fit(X, y)

    return tpot
