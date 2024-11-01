import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.init as init


# 定义判别器
class Discriminator(nn.Module):
    """
    n_channel: 输入图像的通道数
    n_d_feature: 判别器的特征数
    """

    def __init__(self, n_channel=1, n_d_feature=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_d_feature),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * n_d_feature),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * n_d_feature, 1, kernel_size=4),
        )

    def forward(self, x):
        return self.model(x)


# 定义生成器
class Generator(nn.Module):
    """
    n_channel: 输出图像的通道数
    n_g_feature: 生成器的特征数
    latent_size: 输入潜在向量的维度
    """

    def __init__(self, n_channel=1, n_g_feature=32, latent_size=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4, bias=False),
            nn.BatchNorm2d(4 * n_g_feature),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_g_feature),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_g_feature),
            nn.ReLU(),
            nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 初始化权重
def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)


weights_init(Discriminator)
weights_init(Generator)


if __name__ == "__main__":
    # 检查模型并输出参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(summary(Discriminator().to(device), (1, 32, 32)))
    print(summary(Generator().to(device), (32, 1, 1)))
