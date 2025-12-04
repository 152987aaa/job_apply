# import matplotlib.pyplot as plt
# import paddle
#
# # 构造x轴数据
# x = paddle.linspace(-5, 5, 200, "float32")
# # 创建ReLU激活函数层
# relu = paddle.nn.ReLU()
# # 执行计算
# y = relu(x)
# # 输出打印
# plt.plot(x.numpy(), y.numpy(), c="red")
# plt.savefig("result.png")
import matplotlib.pyplot as plt
import paddle
# 构建X轴数据
x = paddle.linspace(-5,5,200,"float32")
# 创建Relu激活函数层
relu = paddle.nn.ReLU()
#执行计算
y = relu(x)
#输出打印
plt.plot(x.numpy(),y.numpy(),c="yellow")
plt.savefig("relu.png")