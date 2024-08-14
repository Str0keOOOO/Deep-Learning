import numpy as np
import matplotlib.pyplot as plt
import torch
from model import CNN
from data import test_loader

plt.rcParams["font.family"] = "SimHei"  # 解决matplotlib中文显示乱码的问题
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题

MODEL_PATH = "./Sequence-Learing/Regression-CNN/model_trained.pt"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except ImportError:
        print("需要先训练模型再进行预测!!!")
    model.to(device)
    model.eval()
    all_b_y = np.array([])
    all_pre_lab = np.array([])
    eval_corrects = 0
    eval_num = 0
    plt.figure(figsize=(8, 6))
    with torch.no_grad():  # 确保不会进行反向传播计算梯度，节省内存和计算资源
        for step, (b_x, b_y) in enumerate(test_loader):
            b_x = b_x.view(-1, 1, 246).float().to(device)
            b_y = b_y.float().to(device)
            output = model(b_x)  # 模型输出
            eval_num += b_x.size(0)
            plt.scatter(b_y.cpu().numpy(), output.cpu().numpy())
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title("预测值与真实值的关系图")
    plt.grid()
    plt.plot([0, 0.2], [0, 0.2], color="r", linestyle="--")  # 绘制一条直线作为基准
    plt.show()
