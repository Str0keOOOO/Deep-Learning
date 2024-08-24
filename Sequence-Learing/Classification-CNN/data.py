import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split

DF_PATH = "./Sequence-Learing/Classification-CNN/data_csv/train.csv"
SIZE = 224
BATCH_SIZE = 24
NUM_WORKERS = 2

# 加载原始数据
df_train = pd.read_csv(DF_PATH)


class MyDataset(Dataset):
    def __init__(self):
        self.features = np.array(
            df_train["heartbeat_signals"].apply(
                lambda x: np.array(list(map(float, x.split(","))), dtype=np.float32)
            )
        )
        self.labels = np.array(
            df_train["label"].apply(lambda x: float(x)), dtype=np.float32
        )

    # 返回数据集大小
    def __len__(self):
        return len(self.features)

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

if __name__ == "__main__":
    # 查看训练和测试数据的前五条
    print(df_train.head())
    print("\n")
    # 检查数据是否有NAN数据
    print(df_train.isna().sum())
    # 确认标签的类别及数量
    print(df_train["label"].value_counts())
    # 查看训练数据集特征
    print(df_train.describe())
    # 查看数据集信息
    print(df_train.info())
    # 绘制每种类别的折线图
    ids = []
    for id, row in df_train.groupby("label").apply(lambda x: x.iloc[2]).iterrows():
        ids.append(int(id))
        signals = list(map(float, row["heartbeat_signals"].split(",")))
        sns.lineplot(data=signals)
    plt.legend(ids)
    plt.show()
