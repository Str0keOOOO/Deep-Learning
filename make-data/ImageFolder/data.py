import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split

PATH = "./make-data/ImageFolder"
SIZE = 224
BATCH_SIZE = 24
NUM_WORKERS = 2

transform = transforms.Compose(
    [
        transforms.Resize([SIZE, SIZE]),
        transforms.ToTensor(),
        # transforms.Grayscale(),
        # transforms.Normalize(mean=[0.5], std=[0.5]),
        # 三通道运行下面的
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data = ImageFolder(PATH, transform=transform)

# 根据分的文件夹的名字来确定的类别
print(data.classes)
# 按顺序为这些类别定义索引为0,1...
print(data.class_to_idx)
# 所有图片的路径和对应的label
print(data.imgs)
# 打印特定的图片
print(data[0][0])

class_label = data.classes
if __name__ == "__main__":
    loder = DataLoader(
        dataset=data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    # 第一个批次数据
    for step, (b_x, b_y) in enumerate(loder):
        if step > 0:
            break
    # matplotlib彩色图像通道不一样[N, channel, height, width]转换为[width, height, channel]
    batch_x = b_x.permute(0, 2, 3, 1).squeeze().numpy()
    batch_y = b_y.numpy()
    print(f"所有类别{class_label}")
    print(f"第{step}个批次的数据{batch_x, batch_y}")

    # 可视化
    plt.figure(figsize=(12, 5))
    for ii in np.arange(len(batch_y)):
        plt.subplot(4, 16, ii + 1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[ii]], size=10)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()
