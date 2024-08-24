import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from model import LSTM
from data import train_loader

LR = 0.01
EPOCHS = 100
MODEL_PATH = "./Sequence-Learing/Regression-RNN-LSTM-GRU/model_trained.pt"


if __name__ == "__main__":
    model = LSTM()
    # 我们要确保的是模型中用到数据要和模型在一个device（GPU或者CPU）上面，其他的数据不需要管。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # 均方误差损失函数
    criterion = nn.MSELoss()
    # 模型送入训练设备
    model = model.to(device)
    train_loss_all = np.array([])
    best_loss = float("inf")
    for epoch in range(EPOCHS):
        print("-" * 20)
        print(f"{epoch+1}/{EPOCHS}")
        # 设置训练模式
        model.train()
        train_loss = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)  # 模型输出
            loss = criterion(output, b_y)  # 计算损失函数
            optimizer.zero_grad()  # 将梯度初始化为0
            loss.backward()  # 反向传播计算
            optimizer.step()  # 更新参数
            # 结果输出
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)

        print(f"train_loss:{train_loss / train_num}")
        train_loss_all = np.append(train_loss_all, train_loss / train_num)

        # 寻找最低损失
        if train_loss_all[-1] < best_loss:
            best_loss = train_loss_all[-1]
        best_model_wts = model.state_dict()

    # 结果
    torch.save(best_model_wts, MODEL_PATH)
    print(f"best_loss:{best_loss}")
    # 画图
    plt.figure(figsize=(6, 5))
    plt.title("Loss")
    plt.plot(
        train_loss_all,
        color="red",
        alpha=0.5,
        linestyle="--",
        linewidth=3,
        marker="o",
        markeredgecolor="red",
        markersize="3",
        markeredgewidth=4,
        label="train_loss",
    )
    plt.legend()
    plt.grid()
    plt.show()
