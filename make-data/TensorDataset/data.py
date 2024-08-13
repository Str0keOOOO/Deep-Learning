import torch
from torch.utils.data import DataLoader, TensorDataset

# 假设我们有一些输入数据 X 和标签 Y
X = torch.randn(100, 3)  # 100个样本，每个样本3个特征
Y = torch.randn(100, 1)  # 100个样本的标签

# 创建 TensorDataset
dataset = TensorDataset(X, Y)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 迭代 DataLoader
for i, (x, y) in enumerate(dataloader):
    print(f"Batch {i}:")
    print(f"Features: {x.size()}, Labels: {y.size()}")
    # 在这里，x 和 y 将是批次的特征和标签
