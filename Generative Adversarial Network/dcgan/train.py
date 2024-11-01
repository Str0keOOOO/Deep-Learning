import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.autograd import Variable
from model import Discriminator, Generator
from data import data_loader

LR = 0.0002
EPOCHS = 5

if __name__ == "__main__":
    discriminator = Discriminator()
    generator = Generator()
    # 我们要确保的是模型中用到数据要和模型在一个device（GPU或者CPU）上面，其他的数据不需要管。
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义优化器generator
    doptimizer = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    goptimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    # 交叉熵损失函数
    criterion = nn.BCEWithLogitsLoss().cuda()
    # 模型送入训练设备
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    for epoch in range(EPOCHS):
        for batch_idx, data in enumerate(data_loader):
            real_images, _ = data
            batch_size = real_images.shape[0]
            # 训练判别器D
            labels = torch.ones(batch_size)  # 真实数据的标签：1
            preds = discriminator(Variable(real_images.type(Tensor)))  # 将真实数据喂给D网络
            outputs = preds.reshape(-1)  # 转换成未知行
            dloss_real = criterion(outputs, labels.type(Tensor))
            dmean_real = outputs.sigmoid().mean()  # 计算判别器将多少真数据判别为真，仅用于输出显示

            noises = torch.randn(batch_size, 32, 1, 1)
            fake_images = generator(noises.type(Tensor))  # 生成假数据
            labels = torch.zeros(batch_size)  # 生成假数据的标签：0
            fake = fake_images.detach()  # 类似于固定生成器参数
            preds = discriminator(fake)  # 将假数据喂给判别器
            outputs = preds.reshape(-1)  # 转换成未知行
            dloss_fake = criterion(outputs.type(Tensor), labels.type(Tensor))
            dmean_fake = outputs.sigmoid().mean()  # 计算判别器将多少假数据判断为真，仅用于输出显示

            dloss = dloss_real + dloss_fake  # 总的鉴别器损失为两者之和
            discriminator.zero_grad()  # 梯度清零
            dloss.backward()  # 反向传播
            doptimizer.step()

            # 训练生成器G
            labels = torch.ones(batch_size)  # 在训练生成器G时，希望生成器的标签为1
            preds = discriminator(fake_images)  # 让假数据通过鉴别网络
            outputs = preds.reshape(-1)  # 转换成未知行
            gloss = criterion(outputs.type(Tensor), labels.type(Tensor))
            gmean_fake = outputs.sigmoid().mean()  # 计算判别器将多少假数据判断为真，仅用于输出显示

            generator.zero_grad()  # 梯度清零
            gloss.backward()  # 反向传播
            goptimizer.step()

            # 输出本步训练结果
            print(
                f"[{epoch+1}/{EPOCHS}]"
                + f"[{batch_idx+1}/{len(data_loader)}]"
                + f"鉴别器G损失:{dloss} 生成器D损失：{gloss}"
                + f"真数据判真比例：{dmean_real} 假数据判真比例：{dmean_fake}/{gmean_fake}"
            )
            if batch_idx % 100 == 0:
                fake = generator(torch.randn(64, 32, 1, 1).cuda())  # 噪声生成假数据
                path = f"./Generative Adversarial Network/dcgan/data_new/gpu{epoch+1}_batch{batch_idx+1}.png"
                print(fake.size())
                save_image(fake, path, normalize=False)
