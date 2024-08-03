import torch
import torch.nn as nn
from torchsummary import summary


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=3, hidden_size=64, num_layers=2, batch_first=True
        )
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        # 需要输入的向量维度是三维（即形状为 (batch_size, sequence_length, 3)）
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 使用最后一个时间步的输出
        return out


if __name__ == "__main__":
    # 检查模型并输出参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM().to(device)
    # print(summary(model, (64, 100, 3, 3)))
