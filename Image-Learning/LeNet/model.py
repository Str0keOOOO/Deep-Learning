import torch
import torch.nn as nn
from torchsummary import summary


class Lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )

    def forward(self, input):
        output = self.model(input)
        return output


if __name__ == "__main__":
    # 检查模型并输出参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lenet().to(device)
    print(summary(model, (1, 28, 28)))
