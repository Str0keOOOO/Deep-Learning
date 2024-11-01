import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

SIZE = 32
BATCH_SIZE = 64
NUM_WORKERS = 2
DATA_PATH = "./Generative Adversarial Network/dcgan/data"

data_trained = MNIST(
    root=DATA_PATH,
    train=True,
    transform=transforms.Compose(
        [
            transforms.Resize(size=SIZE),
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ],
    ),
    download=True,
)


data_loader = DataLoader(dataset=data_trained, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

class_label = data_trained.classes


if __name__ == "__main__":
    data_loader = DataLoader(dataset=data_trained, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # 第一个批次数据
    for step, (b_x, b_y) in enumerate(data_loader):
        if step > 0:
            break
    batch_x = b_x.squeeze().numpy()
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
