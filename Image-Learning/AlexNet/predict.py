import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn import metrics
from model import AlexNet
from data import test_loader, class_label

MODEL_PATH = "./AlexNet/model_trained.pt"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet()
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
    with torch.no_grad():  # 确保不会进行反向传播计算梯度，节省内存和计算资源
        for step, (b_x, b_y) in enumerate(test_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)  # 模型输出
            pre_lab = torch.argmax(output, dim=1)  # softmax标签值
            # 结果输出
            eval_corrects += torch.sum(pre_lab == b_y)
            eval_num += b_x.size(0)
            all_b_y = np.append(all_b_y, b_y.cpu())
            all_pre_lab = np.append(all_pre_lab, pre_lab.cpu())
    print(f"测试集正确率为{eval_corrects.double().item()/eval_num}")

    cm = metrics.confusion_matrix(all_b_y, all_pre_lab)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        cmap="Blues",
        annot=True,
        fmt="d",
        xticklabels=class_label,
        yticklabels=class_label,
    )

    plt.show()
