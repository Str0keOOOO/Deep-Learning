import torch
import torch.nn as nn
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        self.output_size = output_size
        self.relu = nn.ReLU()
        # 第一部分：多个卷积层、批归一化和激活
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 第二部分：更多卷积层和池化
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 第三部分：卷积层和全局平均池化
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )

        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, self.output_size)

    def forward(self, x):
        # 前向传播
        x = self.conv_block1(x)
        x = self.pool1(x)

        x = self.conv_block2(x)
        x = self.pool2(x)

        x = self.conv_block3(x)
        x = self.pool3(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(10).to(device)  # output_size可以根据需要调整
    print(summary(model, (1, 1000)))  # 输入数据的尺寸(1, 1000)根据实际情况调整
