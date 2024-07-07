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

## 图像数据

```powershell
ls Deep-Learn
ls Image-Learning
```

## 构建自己的数据集

```powershell
ls Deep-Learn
ls have-own-data
```

## gpu-test

测试构建DataLoader的最适合num_workers

```powershell
ls Deep-Learn
ls gpu-test
```

