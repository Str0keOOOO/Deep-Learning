import torch
from torchviz import make_dot
from torchvision.models import vgg16  # 以 vgg16 为例

x = torch.randn(4, 3, 32, 32)  # 随机生成一个张量
model = vgg16()  # 实例化 vgg16，网络可以改成自己的网络
out = model(x)  # 将 x 输入网络
g = make_dot(out)  # 实例化 make_dot
g.view()  # 直接在当前路径下保存 pdf 并打开
# g.render(filename='netStructure/myNetModel', view=False, format='pdf')  # 保存 pdf 到指定路径不打开