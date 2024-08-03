import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# BATCH_SIZE的设置尽可能大一些 过小的BATCH_SIZE可能会导致不收敛,此时也可以用减小全连接层来让网络收敛，同时也可以减小param
SIZE = 224
BATCH_SIZE = 24
NUM_WORKERS = 0
DATA_PATH = "./Image-Learning/ResNet18/data"

data_trained = FashionMNIST(
    root=DATA_PATH,
    train=True,
    transform=transforms.Compose(
        [transforms.Resize(size=SIZE), transforms.ToTensor()],
    ),
    download=True,
)

lengths = [
    round(0.8 * len(data_trained)),
    len(data_trained) - round(0.8 * len(data_trained)),
]

train_data, eval_data = random_split(data_trained, lengths)
train_loader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
eval_loader = DataLoader(
    dataset=eval_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

data_test = FashionMNIST(
    root=DATA_PATH,
    train=False,
    transform=transforms.Compose(
        [transforms.Resize(size=SIZE), transforms.ToTensor()],
    ),
    download=True,
)

test_loader = DataLoader(
    dataset=data_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

class_label = data_trained.classes


if __name__ == "__main__":
    loader = DataLoader(
        dataset=data_trained,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    # 第一个批次数据
    for step, (b_x, b_y) in enumerate(loader):
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
