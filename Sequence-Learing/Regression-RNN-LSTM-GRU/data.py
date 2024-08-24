import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

TIME_STEP = 8
BATCH_SIZE = 1
NUM_WORKERS = 0

# 读入数据,以闭市价作为股价构成标签
sample_trained = pd.read_csv(
    "./Sequence-Learing/Regression-RNN-LSTM-GRU/data_csv/train.csv"
).loc[:, "close"]
sample_tested = pd.read_csv(
    "./Sequence-Learing/Regression-RNN-LSTM-GRU/data_csv/val.csv"
).loc[:, "close"]


# 时间序列时间步长为8，通过前8个数据预测第9个
def time_split(data):
    x, y = [], []
    for i in range(len(data) - TIME_STEP):
        x.append([a for a in data[i : i + TIME_STEP]])
        y.append([data[i + TIME_STEP]])
    x = torch.tensor(x).reshape(-1, TIME_STEP, 1)
    y = torch.tensor(y).float().reshape(-1, 1)
    return x, y


# 构造迭代器，返回
data_trained = TensorDataset(*time_split(sample_trained))
data_tested = TensorDataset(*time_split(sample_tested))


train_loader = DataLoader(
    dataset=data_trained, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    dataset=data_tested, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

if __name__ == "__main__":
    loader = DataLoader(
        dataset=data_trained,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    for step, (b_x, b_y) in enumerate(loader):
        if step > 0:
            break
    print(b_x.shape, b_y.shape)
