import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model import CNN

TEST_PATH = "./Sequence-Learing/Classification-CNN/test.csv"
MODEL_PATH = "./Sequence-Learing/Classification-CNN/model_trained.pt"

df_test = pd.read_csv(TEST_PATH)


class MyDataset(Dataset):
    def __init__(self):
        self.features = np.array(
            df_test["heartbeat_signals"].apply(
                lambda x: np.array(list(map(float, x.split(","))), dtype=np.float32)
            )
        )
        self.ids = np.array(df_test["id"], dtype=np.int64)

    # 返回数据集大小
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.ids[index]


data_tested = MyDataset()
test_loader = DataLoader(data_tested, batch_size=1, shuffle=False)

if __name__ == "__main__":
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except ImportError:
        print("需要先训练模型再进行预测!!!")
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        for step, (b_x, id) in enumerate(test_loader):
            input = b_x
            input = input.to(device)
            output = model(input)  # 模型预测输出
            pre_lab = torch.argmax(output, dim=1)  # softmax标签值
            print(f"id:{id.item()}的预测值为:{pre_lab.item()}")
