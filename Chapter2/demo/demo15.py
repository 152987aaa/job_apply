# import paddle
#
# # 创建输入数据
# paddle.seed(102) # 固定随机数，使结果可复现
# x = paddle.randn((3, 2), dtype="float32")
# # 构造两个线性变换层
# myLayer1 = paddle.nn.Linear(2, 4, bias_attr=True)
# myLayer2 = paddle.nn.Linear(4, 5, bias_attr=True)
# # 使用两个线性变换层逐个对x进行计算
# y1 = myLayer1(x)
# y2 = myLayer2(y1)
# print(y2)
# # 将两个线性变换层组合成新的层，再对x进行计算
# myLayer3 = paddle.nn.Sequential(myLayer1, myLayer2)
# y3 = myLayer3(x)
# print(y3)
import paddle
from numpy import dtype

#创建输入数据
paddle.seed(100)
x = paddle.randn((3,2),dtype='float32')
#构建两个线性变换曾
Layer1 = paddle.nn.Linear(2,4,bias_attr=True)
Layer2 = paddle.nn.Linear(4,5,bias_attr=True)
y1 = Layer1(x)
y2 = Layer2(y1)
print(y2)
Layer3 = paddle.nn.Sequential(Layer1,Layer2)
y3 = Layer3(x)
print(y3)