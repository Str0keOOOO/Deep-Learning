import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

BATCH_SIZE = 50
NUM_WORKERS = 0
DATA_PATH = "./Sequence-Learing/Regression-CNN/data.csv"


class MyDataset(Dataset):
    def __init__(self):
        data = pd.read_csv(DATA_PATH, header=None)
        self.features = np.array(data.iloc[:, :-1])
        self.labels = np.array(data.iloc[:, -1]).ravel()

    # 返回数据集大小
    def __len__(self):
        return len(self.features)

    # 返回数据集中第index个样本的特征和标签
    def __getitem__(self, index):
        return self.features[index], self.labels[index]


data_trained = MyDataset()

lengths_1 = [
    round(0.7 * len(data_trained)),
    len(data_trained) - round(0.7 * len(data_trained)),
]
train_data, eval_data = random_split(data_trained, lengths_1)

lengths_2 = [
    round(0.9 * len(eval_data)),
    len(eval_data) - round(0.9 * len(eval_data)),
]

eval_data, test_data = random_split(eval_data, lengths_2)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
eval_loader = DataLoader(
    dataset=eval_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


if __name__ == "__main__":
    loader = DataLoader(
        dataset=data_trained,
        batch_size=10,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    # 第一个批次数据
    for step, (b_x, b_y) in enumerate(loader):
        if step > 0:
            break
    print(b_x.shape, b_y.shape)
