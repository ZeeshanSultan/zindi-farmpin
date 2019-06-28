import os
import numpy as np
import pandas as pd

from glob import glob

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

from sklearn.metrics import log_loss

from config import processed_data_dir, interim_data_dir

seed = 2019

import warnings

warnings.simplefilter('ignore')


def save_probs(clf, ts, X_test, name='train', labels=None):
    # Create output filename
    f_out = os.path.join(interim_data_dir, f'baseline_probs/{name}/{ts}.csv')

    # Predict probabilities
    probs = clf.predict_proba(X_test)

    # Create probabilities dataframe
    probs_df = pd.DataFrame(probs, columns=[f'crop_id_{i}' for i in clf.classes_])

    # Create correct index
    probs_df['Field_Id'] = X_test.index.values
    probs_df.set_index('Field_Id', inplace=True)

    # Add labels to training data
    if labels is not None:
        probs_df['Crop_Id_Ne'] = labels.values

    probs_df.to_csv(f_out)


def make_probs(train_list, test_list):
    for train_fp, test_fp in zip(train_list, test_list):
        ts = os.path.basename(train_fp).split('.')[0]

        print(ts)

        train_dataset = pd.read_csv(train_fp, index_col=0, na_values=-1).dropna(how='all')
        test_dataset = pd.read_csv(test_fp, index_col=0)

        target_col = 'Crop_Id_Ne'

        X_train = train_dataset.drop(target_col, axis=1)
        y_train = train_dataset[target_col]

        X_test = test_dataset.copy()

        pipeline = make_pipeline(
            StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True,
                                                             criterion="entropy",
                                                             max_features=0.3,
                                                             min_samples_leaf=14,
                                                             min_samples_split=11,
                                                             n_estimators=100)
                              ),
            LogisticRegression(C=10.0, dual=False, penalty="l2")
        )

        pipeline.fit(X_train, y_train)

        save_probs(pipeline, ts, X_train, name='train', labels=y_train)
        save_probs(pipeline, ts, X_test, name='test')


if __name__ == '__main__':
    train_files = glob(os.path.join(processed_data_dir, 'baseline/train/*.csv'))
    test_files = glob(os.path.join(processed_data_dir, 'baseline/test/*.csv'))

    make_probs(train_files, test_files)
