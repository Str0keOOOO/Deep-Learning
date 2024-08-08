import torch
import torch.nn as nn
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_unit = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(
                in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=11, stride=1, padding=5
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(4),
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.1),
        )
        self.dense_unit = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs):
        inputs = inputs.view(inputs.size()[0], 1, inputs.size()[1])
        inputs = self.conv_unit(inputs)
        inputs = inputs.view(inputs.size()[0], -1)
        inputs = self.dense_unit(inputs)
        return inputs


if __name__ == "__main__":
    # 检查模型并输出参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    print(summary(model, (205,)))
