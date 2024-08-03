import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

BATCH_SIZE = 32
NUM_WORKERS = 0
DATA_PATH = "./Sequence-Learing/Regression-LSTM/3x2y.csv"

data_pds = pd.read_csv(DATA_PATH)
data = np.array(data_pds.iloc[:, 0:3])
label = np.array(data_pds.iloc[:, 3])
print(data)
data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
label = torch.tensor(label, dtype=torch.float32)
data_trained = TensorDataset(data, label)

lengths = [
    round(0.8 * len(data_trained)),
    len(data_trained) - round(0.8 * len(data_trained)),
]
train_data, eval_data = random_split(data_trained, lengths)

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
    print(b_x.shape, b_y.shape)
    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
    print(f"第{step}个批次的数据{batch_x, batch_y}")
    # 绘制数据分布
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(batch_x[:, 0], batch_x[:, 1], batch_x[:, 2], c=batch_y, s=100, alpha=0.5)
    plt.show()
