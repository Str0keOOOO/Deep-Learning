import torch
import torchvision.models as models

# 使用预定义的 ResNet-18 模型
resnet18 = models.resnet18(pretrained=False)

# 使用预定义的 VGG-16 模型
vgg16 = models.vgg16(pretrained=False)

# 使用预定义的 AlexNet 模型
alexnet = models.alexnet(pretrained=False)

# 使用预定义的 DenseNet-121 模型
densenet121 = models.densenet121(pretrained=False)

# 使用预定义的 Inception v3 模型
inception_v3 = models.inception_v3(pretrained=False)

# 使用预定义的 MobileNet v2 模型
mobilenet_v2 = models.mobilenet_v2(pretrained=False)

# 使用预定义的 ShuffleNet v2 模型
shufflenet_v2 = models.shufflenet_v2_x1_0(pretrained=False)

# 使用预定义的 SqueezeNet 模型
squeezenet = models.squeezenet1_0(pretrained=False)

# 将模型设置为评估模式
resnet18.eval()
vgg16.eval()
alexnet.eval()
densenet121.eval()
inception_v3.eval()
mobilenet_v2.eval()
shufflenet_v2.eval()
squeezenet.eval()

# 打印模型结构
print(resnet18)
print(vgg16)
print(alexnet)
print(densenet121)
print(inception_v3)
print(mobilenet_v2)
print(shufflenet_v2)
print(squeezenet)