import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import GoogLeNet
from data import train_loader, eval_loader

LR = 0.001
EPOCHS = 10
MODEL_PATH = "./Image-Learning/GoogLeNet/model_trained.pt"


if __name__ == "__main__":
    model = GoogLeNet()
    # 我们要确保的是模型中用到数据要和模型在一个device（GPU或者CPU）上面，其他的数据不需要管。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 模型送入训练设备
    model = model.to(device)
    train_loss_all = np.array([])
    eval_loss_all = np.array([])
    train_acc_all = np.array([])
    eval_acc_all = np.array([])
    best_acc = 0
    for epoch in range(EPOCHS):
        print("-" * 20)
        print(f"{epoch+1}/{EPOCHS}")
        # 设置训练模式
        model.train()
        train_loss = 0
        train_corrects = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)  # 模型输出
            pre_lab = torch.argmax(output, dim=1)  # softmax标签值
            loss = criterion(output, b_y)  # 计算损失函数
            optimizer.zero_grad()  # 将梯度初始化为0
            loss.backward()  # 反向传播计算
            optimizer.step()  # 更新参数
            # 结果输出
            # 每一个batch的平均损失值*batch大小相加表示一个epoch的总损失值
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y)
            train_num += b_x.size(0)
        print(f"train_loss:{train_loss / train_num}")
        print(f"train_acc:{train_corrects / train_num}")
        train_loss_all = np.append(train_loss_all, train_loss / train_num)
        train_acc_all = np.append(
            train_acc_all, train_corrects.double().item() / train_num
        )

        # 设置评估模式
        model.eval()
        eval_loss = 0
        eval_corrects = 0
        eval_num = 0
        for step, (b_x, b_y) in enumerate(eval_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)  # 模型输出
            pre_lab = torch.argmax(output, dim=1)  # softmax标签值
            loss = criterion(output, b_y)  # 计算损失函数
            # 结果输出
            eval_loss += loss.item() * b_x.size(0)
            eval_corrects += torch.sum(pre_lab == b_y)
            eval_num += b_x.size(0)
        print(f"eval_loss:{eval_loss / eval_num}")
        print(f"eval_acc:{eval_corrects / eval_num}")
        eval_loss_all = np.append(eval_loss_all, eval_loss / eval_num)
        eval_acc_all = np.append(eval_acc_all, eval_corrects.double().item() / eval_num)
        # 寻找最高精确度
        if eval_acc_all[-1] > best_acc:
            best_acc = eval_acc_all[-1]
            best_model_wts = model.state_dict()

    # 结果
    torch.save(best_model_wts, MODEL_PATH)
    # 画图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("loss_all")
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
        label="train_loss_all",
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
        label="eval_acc_all",
    )
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.title("acc_all")
    plt.plot(
        train_acc_all,
        color="red",
        alpha=0.5,
        linestyle="--",
        linewidth=3,
        marker="o",
        markeredgecolor="red",
        markersize="3",
        markeredgewidth=4,
        label="train_loss_all",
    )
    plt.plot(
        eval_acc_all,
        color="blue",
        alpha=0.5,
        linestyle="--",
        linewidth=3,
        marker="o",
        markeredgecolor="blue",
        markersize="3",
        markeredgewidth=4,
        label="eval_acc_all",
    )
    plt.legend()
    plt.grid()
    plt.show()