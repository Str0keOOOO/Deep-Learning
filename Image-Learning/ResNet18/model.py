import torch
import torch.nn as nn
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super().__init__()
        self.ReLu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
            stride=strides,
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_channels,
                kernel_size=1,
                stride=strides,
            )
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLu(y + x)
        return y


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block2 = nn.Sequential(
            Residual(64, 64, use_1conv=False, strides=1),
            Residual(64, 64, use_1conv=False, strides=1),
        )
        self.block3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, strides=2),
            Residual(128, 128, use_1conv=False, strides=1),
        )
        self.block4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, strides=2),
            Residual(256, 256, use_1conv=False, strides=1),
        )
        self.block5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, strides=2),
            Residual(512, 512, use_1conv=False, strides=1),
        )
        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x


if __name__ == "__main__":
    # 检查模型并输出参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    print(summary(model, (1, 224, 224)))
