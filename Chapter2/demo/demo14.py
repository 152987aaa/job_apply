# import paddle
#
# # 定义线性变换层W和b的初始权重
# weight_attr = paddle.ParamAttr(
#     name="weight", initializer=paddle.nn.initializer.Constant(value=0.5)
# )
# bias_attr = paddle.ParamAttr(
#     name="bias", initializer=paddle.nn.initializer.Constant(value=1.0)
# )
#
# # 创建线性变换层
# linear = paddle.nn.Linear(2, 4, weight_attr=weight_attr, bias_attr=bias_attr)
# print("W:")
# print(linear.weight.numpy())
# print("b:")
# print(linear.bias.numpy())
#
# # 创建输入数据
# x = paddle.randn((3, 2), dtype="float32")
# print("x:")
# print(x.numpy())
#
# # 执行计算
# y = linear(x)
#
# # 输出
# print("y:")
# print(y.numpy())
import paddle
# 定义线性变换层w和b的初始权重
weight_attr = paddle.ParamAttr(
    name="weight",initializer=paddle.nn.initializer.Constant(value=0.5)
)
bias_attr = paddle.ParamAttr(
    name = "bias",initializer=paddle.nn.initializer.Constant(value=1.0)
)
#创建线性变换层
linear = paddle.nn.Linear(2,4,weight_attr=weight_attr,bias_attr=bias_attr)
print("W:")
print(linear.weight.numpy())
print("b:")
print(linear.bias.numpy())
#创建输入数据
x = paddle.randn((3,2),dtype='float32')
print("x:")
print(x.numpy())
y = linear(x)
print("y:")
print(y.numpy())