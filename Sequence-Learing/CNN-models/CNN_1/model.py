import torch
import torch.nn as nn
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        # 假设我们使用较小的卷积核和步长来处理长序列
        # 注意：这些参数可能需要根据你的具体任务进行调整
        self.output_size = output_size
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=64, kernel_size=128, stride=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=64, stride=64)
        # 由于序列很长，我们可以考虑使用全局平均池化来进一步降低维度
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # 添加全连接层进行分类
        self.fc1 = nn.Linear(
            64, self.output_size
        )  # 注意：这里的64是假设经过池化后的特征图数量，实际可能不同

    # 四分类输出

    def forward(self, x):
        # 应用卷积层和激活函数
        x = self.relu(self.conv1(x))
        # 应用池化层
        x = self.pool(x)
        # 应用全局平均池化
        x = self.global_avg_pool(x)
        # 展平（由于全局平均池化，这一步实际上是多余的，但为了与全连接层兼容，我们还是这样做）
        x = x.view(x.size(0), -1)
        # 应用全连接层
        x = self.relu(self.fc1(x))
        return x


if __name__ == "__main__":
    # 检查模型并输出参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(4).to(device)
    print(summary(model, (1, 40960)))
