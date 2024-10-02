import numpy as np
import torch.nn as nn
import torch

embedding_pretrained = torch.tensor(
    np.load("./NLP/emotional-RNN-LSTM-GRU/data/embedding_Tencent.npz")[
        "embeddings"
    ].astype("float32")
)
# 预训练词向量
embed = embedding_pretrained.size(1)  # 词向量维度


# 定义LSTM模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练的词向量模型，freeze=False 表示允许参数在训练中更新
        # 在NLP任务中，当我们搭建网络时，第一层往往是嵌入层，对于嵌入层有两种方式初始化embedding向量，
        # 一种是直接随机初始化，另一种是使用预训练好的词向量初始化。
        self.embedding = nn.Embedding.from_pretrained(
            embedding_pretrained, freeze=False
        )
        self.lstm = nn.LSTM(
            input_size=embed,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )
        self.fc = nn.Linear(128, 2)

        for name, w in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(w)
            elif "bias" in name:
                nn.init.constant_(w, 0)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out
