import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

BATCH_SIZE = 50
NUM_WORKERS = 0
DATA_PATH = "./Sequence-Learing/Classification-RNN-LSTM-GRU/data/"


class MyDataset(Dataset):
    def __init__(self):
        self.features = []
        self.labels = []
        for i in range(6):
            file_path = os.path.join(DATA_PATH, f"{i}")
            for filename in os.listdir(file_path):
                if filename.endswith(".csv"):
                    csv_path = os.path.join(file_path, filename)
                    csv_data = pd.read_csv(csv_path, header=None)
                    csv_data = np.array(csv_data[:480])
                    self.features.append(csv_data)
                    self.labels.append(i)

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

class_label = [
    "蹲姿到站立(右蹲)",
    "蹲姿到站立(左蹲)",
    "行进",
    "原地踏步",
    "站立到蹲姿(右蹲)",
    "站立到蹲姿(左蹲)",
]

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
    color = ["blue", "red", "green", "yellow", "purple", "orange"]
    plt.figure(figsize=(12, 6))
    for i in range(len(b_x)):
        for j in range(4):
            plt.subplot(2, 2, j + 1)
            plt.plot(b_x[i, :, j].numpy(), color=color[b_y[i].item()])
            plt.title(f"Feature {j}")
    plt.show()
