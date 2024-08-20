import torch
import torch.nn as nn
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        self.output_size = output_size
        self.relu = nn.ReLU()
        # 第一层卷积层
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=64, kernel_size=16, stride=1, padding=7
        )
        self.bn1 = nn.BatchNorm1d(64)

        # 第二层卷积层
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=3
        )
        self.bn2 = nn.BatchNorm1d(128)

        # 第三层卷积层
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)

        # 池化层（使用较大尺寸的池化核）
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc1 = nn.Linear(128, self.output_size)

    def forward(self, x):
        # 第一层卷积+激活+池化
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # 第二层卷积+激活+池化
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # 第三层卷积+激活+池化
        x = self.relu(self.bn3(self.conv3(x)))
        # 全局平均池化
        x = self.global_avg_pool(x)
        # 展平层
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    # 检查模型并输出参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(4).to(device)
    print(summary(model, (1, 40960)))
