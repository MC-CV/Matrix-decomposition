import numpy as np


def rmse(score, X_test):
    return np.sqrt(
        1 / (len(score) * len(score[0])) * np.sum((score - X_test) * (score - X_test))
    )

