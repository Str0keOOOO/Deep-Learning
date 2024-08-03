import torch
import numpy as np
from model import LSTM

MODEL_PATH = "./Sequence-Learing/Regression-LSTM/model_trained.pt"

if __name__ == "__main__":
    model = LSTM()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except ImportError:
        print("需要先训练模型再进行预测!!!")
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        # 当data为[2,3,4]时，求预测值
        input = torch.tensor(np.array([2, 3, 4]).reshape(1, 1, 3), dtype=torch.float32)
        input = input.to(device)
        outputs = model(input)  # 模型预测输出
        print(f"预测值为:{outputs.item()}")
