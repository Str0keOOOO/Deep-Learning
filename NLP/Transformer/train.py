import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from model import CopyTaskModel
from data import generate_random_batch

BATCH_SIZE = 2
MAX_LENGTH = 16
LR = 3e-4
EPOCHS = 2000
MODEL_PATH = "./NLP/Transformer/model_trained.pt"


if __name__ == "__main__":
    model = CopyTaskModel()
    # 我们要确保的是模型中用到数据要和模型在一个device（GPU或者CPU）上面，其他的数据不需要管。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # 均方误差损失函数
    criterion = nn.CrossEntropyLoss()
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
        src, tgt, tgt_y, n_tokens = generate_random_batch(
            batch_size=BATCH_SIZE, max_length=MAX_LENGTH
        )
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_y = tgt_y.to(device)
        output = model(src, tgt)  # 进行transformer的计算
        output = model.predictor(output)  # 将结果送给最后的线性层进行预测
        loss = (
            criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_y.contiguous().view(-1),
            )
            / n_tokens
        )  # 计算损失函数
        optimizer.zero_grad()  # 将梯度初始化为0
        loss.backward()  # 反向传播计算
        optimizer.step()  # 更新参数
        # 结果输出
        print(f"train_loss:{loss.item() * BATCH_SIZE}")
        train_loss_all = np.append(train_loss_all, loss.item() * BATCH_SIZE)

        # 寻找最低损失
        if train_loss_all[-1] < best_loss:
            best_loss = train_loss_all[-1]
            best_model_wts = model.state_dict()

    # 结果
    torch.save(best_model_wts, MODEL_PATH)
    print(f"best_loss:{best_loss}")
    # 画图
    plt.figure(figsize=(12, 5))
    plt.title("Loss")
    plt.plot(
        train_loss_all,
        color="red",
        alpha=0.5,
        linestyle="--",
        linewidth=1,
        marker="o",
        markeredgecolor="red",
        markersize="1",
        markeredgewidth=2,
        label="train_loss",
    )
    plt.legend()
    plt.grid()
    plt.show()
