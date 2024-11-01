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

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(32 * 32, 256),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(256, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )

    def forward(self, x):
        x = x.view(-1, 32 * 32)
        return self.model(x)


# 定义生成器
class Generator(nn.Module):
    """
    n_channel: 输出图像的通道数
    n_g_feature: 生成器的特征数
    latent_size: 输入潜在向量的维度
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 32 * 32),  # 线性变换
            nn.Tanh(),  # Tanh激活使得生成数据分布在【-1,1】之间
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 32, 32)
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
    print(summary(Generator().to(device), (1, 1, 100)))
