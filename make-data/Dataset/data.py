import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

DATAPATH = "./make-data/Dataset/written-num/data"
LABELPATH = "./make-data/Dataset/written-num/label/label.txt"
SIZE = 224
BATCH_SIZE = 24
NUM_WORKERS = 2

transform = transforms.Compose(
    [
        transforms.Resize([SIZE, SIZE]),
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        # 三通道运行下面的
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MyDataset(Dataset):
    def __init__(self):
        self.images = []
        for filename in os.listdir(DATAPATH):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(DATAPATH, filename)
                image = Image.open(image_path)
                self.images.append((image, filename))
        self.labels = pd.read_csv(LABELPATH, header=None, index_col=0)

    # 返回数据集大小
    def __len__(self):
        return len(self.images)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        # 需要将图片大小统一并变成tensor形式
        img = transform(self.images[index][0])
        label = int(self.labels[1][self.images[index][1]])
        return img, label


data_trained = MyDataset()
lengths = [
    round(0.8 * len(data_trained)),
    len(data_trained) - round(0.8 * len(data_trained)),
]

train_data, eval_data = random_split(data_trained, lengths)
train_loder = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
eval_loder = DataLoader(
    dataset=eval_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

class_label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

if __name__ == "__main__":
    loder = DataLoader(
        dataset=data_trained,
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
