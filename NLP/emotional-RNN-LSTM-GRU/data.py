import numpy as np
import pickle as pkl
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(42)

BATCH_SIZE = 128
NUM_WORKERS = 0
DATA_PATH = "./NLP/emotional-RNN-LSTM-GRU/data/data.txt"  # 数据集
VOCAB_PATH = "./NLP/emotional-RNN-LSTM-GRU/data/vocab.pkl"  # 词表
EMBEDDING_PATH = (
    "./NLP/emotional-RNN-LSTM-GRU/data/embedding_Tencent.npz"  # 预训练词向量
)
PAD_SIZE = 50  # 每句话处理成的长度(短填长切)
UNK, PAD = "<UNK>", "<PAD>"  # 未知字，padding符号

embedding_pretrained = torch.tensor(np.load(EMBEDDING_PATH)["embeddings"])
vocab = pkl.load(open(VOCAB_PATH, "rb"))
class_label = ["negative", "active"]


class TextDataset(Dataset):
    def __init__(self):
        super().__init__()
        content_processed = []
        with open(DATA_PATH, "r", encoding="gbk") as f:
            for line in tqdm(f):
                # 默认删除字符串line中的空格、’\n’、't’等。
                lin = line.strip()
                if not lin:
                    continue
                label, content = lin.split("	####	")
                # 分割器，分词每个字
                token = [y for y in content]
                if PAD_SIZE:
                    # 如果字长度小于指定长度，则填充，否则截断
                    if len(token) < PAD_SIZE:
                        token.extend([vocab.get(PAD)] * (PAD_SIZE - len(token)))
                    else:
                        token = token[:PAD_SIZE]
                # 将每个字映射为ID
                # 如果在词表vocab中有word这个单词，那么就取出它的id；
                # 如果没有，就去除UNK（未知词）对应的id，其中UNK表示所有的未知词（out of vocab）都对应该id
                # word_line存储每个字的id
                words_line = []
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                content_processed.append((words_line, int(label)))
        self.texts = torch.LongTensor([x[0] for x in content_processed])
        self.labels = torch.LongTensor([x[1] for x in content_processed])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        self.text = self.texts[index]
        self.label = self.labels[index]
        return self.texts[index], self.labels[index]


data_trained = TextDataset()

lengths1 = [
    round(0.6 * len(data_trained)),
    len(data_trained) - round(0.6 * len(data_trained)),
]

train_data, eval_data = random_split(data_trained, lengths1)


lengths2 = [
    round(0.5 * len(eval_data)),
    len(eval_data) - round(0.5 * len(eval_data)),
]

eval_data, test_data = random_split(eval_data, lengths2)

train_loader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
eval_loader = DataLoader(
    dataset=eval_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

# 以上是数据预处理的部分
if __name__ == "__main__":
    loader = DataLoader(
        dataset=data_trained,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    for step, (b_x, b_y) in enumerate(loader):
        if step > 0:
            break
    print(b_x.shape, b_y.shape)
