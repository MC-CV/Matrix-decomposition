import numpy as np
from numpy.linalg import norm
from rmse import rmse
import time

if __name__ == "__main__":
    # 读取数据矩阵
    X_train = np.load("train.npy")
    X_test = np.load("test.npy")

    # 协同过滤
    # 计算每一行向量的范数
    s = time.time()
    norm_train = norm(X_train, axis=1).reshape(-1, 1)
    # 利用矩阵直接计算两两行向量之间的点乘
    mul_dot = np.dot(X_train, X_train.T)
    # 余弦相似度的分母利用numpy的广播机制得到范数矩阵
    fanshu = np.dot(norm_train, norm_train.T)
    # 分别根据定义计算分子和分母
    fenzi = np.dot(mul_dot, X_train) / fanshu
    fenmu = np.sum(mul_dot / fanshu, axis=1)
    # 每一行的除数相同，故将fenmu拉成一列之后利用广播直接相除
    score = fenzi / fenmu.reshape(-1, 1)

    print("总用时 = ", time.time() - s, "s")
    # 打印RMSE
    print(rmse(score, X_test))
