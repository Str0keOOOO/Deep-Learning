import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split

TIME_STEP = 40  # 时间步数 / 图片高度
INPUT_SIZE = 30  # 每步输入值 / 图片每行像素
BATCH_SIZE = 64
NUM_WORKERS = 0


class MyDataset(Dataset):
    def __init__(self):
        sample_0 = pd.read_csv(
            "./Sequence-Learing/Classification-RNN-LSTM/bear_fault/sample/0.csv"
        )
        sample_1 = pd.read_csv(
            "./Sequence-Learing/Classification-RNN-LSTM/bear_fault/sample/1.csv"
        )
        sample_2 = pd.read_csv(
            "./Sequence-Learing/Classification-RNN-LSTM/bear_fault/sample/2.csv"
        )
        sample_3 = pd.read_csv(
            "./Sequence-Learing/Classification-RNN-LSTM/bear_fault/sample/3.csv"
        )
        sample_4 = pd.read_csv(
            "./Sequence-Learing/Classification-RNN-LSTM/bear_fault/sample/4.csv"
        )
        sample = pd.concat([sample_0, sample_1, sample_2, sample_3, sample_4], axis=1)
        self.features = np.array(sample)[:1200].T
        self.labels = np.concatenate(
            (
                np.array([0] * sample_0.shape[1]),
                np.array([1] * sample_1.shape[1]),
                np.array([2] * sample_2.shape[1]),
                np.array([3] * sample_3.shape[1]),
                np.array([4] * sample_4.shape[1]),
            )
        )

    # 返回数据集大小
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


data_trained = MyDataset()
lengths = [
    round(0.8 * len(data_trained)),
    len(data_trained) - round(0.8 * len(data_trained)),
]

train_data, eval_data = random_split(data_trained, lengths)
train_loader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
eval_loader = DataLoader(
    dataset=eval_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)


class_label = ["正常", "内圈故障", "外圈故障", "滚动体故障", "保持架故障"]


if __name__ == "__main__":
    loader = DataLoader(
        dataset=data_trained,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    # 第一个批次数据
    for step, (b_x, b_y) in enumerate(loader):
        if step > 0:
            break
    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
    print(f"所有类别{class_label}")
    print(f"第{step}个批次的数据{batch_x.shape}, {batch_y.shape}")

    # 可视化
    plt.figure(figsize=(12, 5))
    for ii in range(4):
        plt.plot(batch_x[ii], label=class_label[batch_y[ii]])
    plt.title("Bear Fault Classification")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
