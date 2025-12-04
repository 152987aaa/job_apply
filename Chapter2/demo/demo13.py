# import paddle
#
# # 构建输入数据
# input = paddle.uniform([1, 3, 32, 32], dtype="float32", min=-1, max=1)
# # 创建最大池化层
# maxpool2d = paddle.nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
# # 计算
# output = maxpool2d(input)
# # 输出
# print(output.shape)
import paddle
# 构建输入数据
input = paddle.uniform([1,3,32,32],dtype='float32',min=-1,max=1)
# 构建最大池化层
maxpool2d = paddle.nn.MaxPool2D(kernel_size=2,stride=2,padding=0)
#计算
output = maxpool2d(input)
#输出
print(output.shape)
