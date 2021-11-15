import numpy as np
from rmse import rmse
import matplotlib.pyplot as plt

# Frobenius 范数
def frobenius(a):
    return np.sqrt(np.sum(a * a))


class matrix:
    # 初始化
    def __init__(self, rate=0.01, k=100, lamda=0.2):
        self.rate = rate
        self.k = k
        self.U = np.random.random([10000, self.k]) / 100
        self.V = np.random.random([10000, self.k]) / 100
        self.epoch = 10000
        self.lamda = lamda

    # 训练的同时每一轮进行测试计算在测试集上的RMSE
    def train_eval(self, X, test):
        # 指示矩阵A
        A = X > 0
        # 输入0时刻的loss和target
        losses = []
        losses.append(rmse(np.dot(self.U, self.V.T), test))
        targets = []
        targets.append(
            1 / 2 * frobenius((A * (X - np.dot(self.U, self.V.T))))
            + self.lamda * frobenius(self.U)
            + self.lamda * frobenius(self.V)
        )
        for i in range(self.epoch):
            # 上一轮目标的值
            target = targets[i]

            # 预测的矩阵
            pred = np.dot(self.U, self.V.T)

            # 梯度下降更新
            self.U -= self.rate * (
                (np.dot(A * (np.dot(self.U, self.V.T) - X), self.V))
                + 2 * self.lamda * self.U
            )
            self.V -= self.rate * (
                (np.dot((A * (np.dot(self.U, self.V.T) - X)).T, self.U))
                + 2 * self.lamda * self.V
            )

            # 本轮迭代后目标的值
            target_after = (
                1 / 2 * frobenius((A * (X - np.dot(self.U, self.V.T))))
                + self.lamda * frobenius(self.U)
                + self.lamda * frobenius(self.V)
            )

            # 计算在测试集上的RMSE并记录，同时记录target的值
            loss = rmse(pred, test)
            print("epoch:", i + 1, "target:", target_after, "loss:", loss)
            losses.append(loss)
            targets.append(target_after)

            # 判断当target的变化小于等于1的时候作为收敛条件，此时退出迭代
            if abs(target_after - target) <= 1:
                break

        # 画出迭代过程中目标函数值和测试集上RMSE的变化
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        ax.plot(range(1, len(losses)), losses[1:], label="RMSE", color="red")
        ax.set_xlabel("epochs")
        ax.set_ylabel("RMSE on testset")
        ax2.plot(range(1, len(losses)), targets[1:], label="target", color="blue")
        ax2.set_ylabel("target")
        ax.legend(loc=1)
        ax2.legend(loc=2)
        plt.title(
            "k="
            + str(self.k)
            + ",lambda="
            + str(self.lamda)
            + ",alpha="
            + str(self.rate)
        )
        plt.show()
        print("最终的RMSE:", losses[-1])


if __name__ == "__main__":
    # 读取数据矩阵
    X_train = np.load("train.npy")
    X_test = np.load("test.npy")
    # 初始化参数设置
    model = matrix(rate=0.0001, k=100, lamda=0.01)
    # 开始运行
    model.train_eval(X_train, X_test)
