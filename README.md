# 深度学习模板

- Python 3.9.18
- torch 2.1.1+cu121
- torchsummary 1.5.1

  ```mermaid
    graph LR
      深度学习-->  图像数据
      深度学习-->  时序数据
     	图像数据-->  LeNet
     	图像数据-->  AlexNet
     	图像数据-->  VGG
     	图像数据-->  GoogLeNet
     	图像数据-->  ResNet
     	时序数据-->  RNN
     	时序数据--> LSTM
     	时序数据--> CNN
  ```

## 图像学习
torchModel为torch.nn.Module的子类，可继承该类实现自己的模型

## 信号学习 --by Dragon 
Sequence-Learing为信号学习

## 构建自己的数据集

两种方法

## gpu-test

测试构建DataLoader的最适合num_workers

## visualization

可视化方法

## 注意

### 关于路径

代码报错的第一可能是因为相对路径错误，请查看是否在指定的正确工作区或者自行修改路径参数

### 显存上溢

请修改batch_size
