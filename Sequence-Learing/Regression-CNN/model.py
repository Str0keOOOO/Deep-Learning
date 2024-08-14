import torch
import torch.nn as nn
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv8 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.maxpool(x)
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # 检查模型并输出参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    print(summary(model, (1, 246)))
