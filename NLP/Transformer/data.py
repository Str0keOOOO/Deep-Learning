import random
import torch

# 可以理解为生成一个中译中的随机句子来训练模型，src->tgt
def generate_random_batch(batch_size, max_length=16):
    src = []
    for i in range(batch_size):
        # 随机生成句子长度
        random_len = random.randint(1, max_length - 2)
        # 随机生成句子词汇，并在开头和结尾增加<bos>和<eos>
        random_nums = [0] + [random.randint(3, 9) for _ in range(random_len)] + [1]
        # 如果句子长度不足max_length，进行填充
        random_nums = random_nums + [2] * (max_length - random_len - 2)
        src.append(random_nums)
    src = torch.LongTensor(src)
    # tgt不要最后一个token
    tgt = src[:, :-1]
    # tgt_y不要第一个的token
    tgt_y = src[:, 1:]
    # 计算tgt_y，即要预测的有效token的数量
    n_tokens = (tgt_y != 2).sum()
    # 这里的n_tokens指的是我们要预测的tgt_y中有多少有效的token，后面计算loss要用
    return src, tgt, tgt_y, n_tokens


if __name__ == "__main__":
    src, tgt, tgt_y, n_tokens = generate_random_batch(batch_size=2, max_length=6)
    print(src)
    print(tgt)
    print(tgt_y)
    print(n_tokens)
