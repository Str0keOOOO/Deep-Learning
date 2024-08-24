import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import LSTM
from data import test_loader


MODEL_PATH = "./Sequence-Learing/Regression-RNN-LSTM-GRU/model_trained.pt"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM()
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except ImportError:
        print("需要先训练模型再进行预测!!!")
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    Val_y, Val_predict = [], []
    for step, (b_x, b_y) in enumerate(test_loader):
        train_loss = 0
        train_num = 0
        with torch.no_grad():
            b_x = b_x.cuda()
            b_y = b_y.cpu()
            output = model(b_x).cpu()
            loss = criterion(output, b_y)  # 计算损失函数
            Val_y.append(b_y.item())
            Val_predict.append(output.item())
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
    print("均方误差:", train_loss / train_num)
    fig = plt.figure(figsize=(8, 5))
    # 红色表示真实值，绿色表示预测值
    plt.plot(Val_y, linestyle="--", color="r")
    plt.plot(Val_predict, color="g")
    plt.title("stock price")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.show()
