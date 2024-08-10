import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(  # LSTM效果要比nn.RNN()好多了,如果想要用RNN就将LSTM改成RNN即可
            input_size=28,  # 图片每行的数据28像素点
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 10)  # 输出层

    def forward(self, x):
        r_out, _ = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])  # 选取最后一个时间点的 r_out 输出
        return out


if __name__ == "__main__":
    # 检查模型并输出参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM().to(device)
    out = model(torch.randn(64, 28, 28).to(device))
    print(out.size())
