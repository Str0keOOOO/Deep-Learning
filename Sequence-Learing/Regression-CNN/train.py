import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import CNN
from data import train_loader, eval_loader

LR = 0.0001
EPOCHS = 1000
MODEL_PATH = "./Sequence-Learing/Regression-CNN/model_trained.pt"


if __name__ == "__main__":
    model = CNN()
    # 我们要确保的是模型中用到数据要和模型在一个device（GPU或者CPU）上面，其他的数据不需要管。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # 均方误差损失函数
    criterion = nn.MSELoss()
    # 模型送入训练设备
    model = model.to(device)
    train_loss_all = np.array([])
    eval_loss_all = np.array([])
    best_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"{epoch+1}/{EPOCHS}")
        # 设置训练模式
        model.train()
        train_loss = 0
        train_num = 0
        for b_x, b_y in train_loader:
            b_x = b_x.view(-1, 1, 246).float().to(device)
            b_y = b_y.float().to(device)
            output = model(b_x)  # 模型输出
            loss = criterion(output, b_y.unsqueeze(1))  # 计算损失函数
            optimizer.zero_grad()  # 将梯度初始化为0
            loss.backward()  # 反向传播计算
            optimizer.step()  # 更新参数
            # 结果输出
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)

        print(f"train_loss:{train_loss / train_num}")
        train_loss_all = np.append(train_loss_all, train_loss / train_num)

        # 设置评估模式
        model.eval()
        eval_loss = 0
        eval_num = 0
        with torch.no_grad():
            for b_x, b_y in eval_loader:
                b_x = b_x.view(-1, 1, 246).float().to(device)
                b_y = b_y.float().to(device)
                output = model(b_x)  # 模型输出
                loss = criterion(output, b_y.unsqueeze(1))  # 计算损失函数
                eval_loss += loss.item() * b_x.size(0)
                eval_num += b_x.size(0)

        print(f"eval_loss:{eval_loss / eval_num}")
        print("-" * 20)
        eval_loss_all = np.append(eval_loss_all, eval_loss / eval_num)

        # 寻找最低损失
        if eval_loss_all[-1] < best_loss:
            best_loss = eval_loss_all[-1]
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
    plt.plot(
        eval_loss_all,
        color="blue",
        alpha=0.5,
        linestyle="--",
        linewidth=3,
        marker="o",
        markeredgecolor="blue",
        markersize="3",
        markeredgewidth=4,
        label="eval_loss",
    )
    plt.legend()
    plt.grid()
    plt.show()
