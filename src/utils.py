import os
import pandas as pd
from config import raw_data_dir


def make_sub(probs):
    test_sub = pd.read_csv(os.path.join(raw_data_dir, 'sample_submission.csv'), index_col=0)
    probs_df = pd.DataFrame(probs, columns=list(test_sub))
    probs_df.index = test_sub.index.values
    return probs_df
