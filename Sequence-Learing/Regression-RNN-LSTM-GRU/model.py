import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1,  # 输入特征维度,当前特征为股价，维度为1
            hidden_size=20,  # 隐藏层神经元个数，或者也叫输出的维度
            num_layers=1,
        )
        self.out = nn.Linear(20, 1)

    def forward(self, x):
        r_out, _ = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])  # 选取最后一个时间点的 r_out 输出
        return out


if __name__ == "__main__":
    # 检查模型并输出参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM().to(device)
    out = model(torch.randn(1, 8, 1).to(device))
    print(out.size())
