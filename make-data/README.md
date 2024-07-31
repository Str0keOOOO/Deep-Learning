# 构建自己的数据集

## 方法一：制作Dataset

### 注意

预测集没有label，可以在Dataset中return对应的名字，这样就可以把预测的名字和相应预测标签对应起来

## 方法二：制作ImageFolder

需要将图片保存为形如以下的结构：

```mkdir
make-data/ImageFolder/
│
├── cat/
│   ├── 1.jpg
│   └── 2.jpg
│
└── dog/
    ├── 1.jpg
    └── 2.jpg
```
