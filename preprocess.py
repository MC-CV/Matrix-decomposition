import numpy as np
from tqdm import tqdm
from collections import defaultdict


def preprocess(tp: str):
    # 加载user的id,并用字典进行标号（空间换时间，后续不用寻找索引）
    user_ids = defaultdict(int)
    idx = 0
    with open("data/users.txt") as f:
        for line in f.readlines():
            s = line.strip()
            user_ids[s] = idx
            idx += 1

    # 初始化res用以保存最终结果（此处用numpy建立否则第二题无法加速运算）
    res = np.zeros([10000, 10000])
    path = "data/netflix" + "_" + tp + ".txt"

    # 读取训练集/测试集中每一行数据并按照对应索引存入res
    with open(path) as f:
        for line in tqdm(f.readlines()):
            s = line.strip().split(" ")
            res[user_ids[s[0]]][int(s[1]) - 1] = int(s[2])
    return res


if __name__ == "__main__":
    # 分别预处理得到训练集与测试集
    trainset = preprocess("train")
    np.save("train.npy", trainset)
    testset = preprocess("test")
    np.save("test.npy", testset)

