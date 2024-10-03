import torch
from model import CopyTaskModel
from train import MAX_LENGTH

MODEL_PATH = "./NLP/Transformer/model_trained.pt"

if __name__ == "__main__":
    model = CopyTaskModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except ImportError:
        print("需要先训练模型再进行预测!!!")
    with torch.no_grad():
        # 随便定义一个src
        src = torch.LongTensor([[0, 4, 3, 4, 6, 8, 9, 9, 8, 1, 2, 2]])
        # tgt从<bos>开始，看看能不能重新输出src中的值
        tgt = torch.LongTensor([[0]])
        src = src.to(device)
        tgt = tgt.to(device)
        # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
        for i in range(MAX_LENGTH):
            # 进行transformer计算
            out = model(src, tgt)
            # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
            predict = model.predictor(out[:, -1])
            # 找出最大值的index
            y = torch.argmax(predict, dim=1)
            # 和之前的预测结果拼接到一起
            tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
            # 如果为<eos>，说明预测结束，跳出循环
            if y == 1:
                break
        print(tgt)
