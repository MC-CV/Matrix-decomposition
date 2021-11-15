import numpy as np

# 测试集掩膜
X_test = np.load("test.npy")
mask = X_test > 0

# 根据定义计算RMSE
def rmse(score, X_test):
    score = mask * score
    return np.sqrt(
        1 / (np.sum(X_test > 0)) * np.sum((score - X_test) * (score - X_test))
    )

