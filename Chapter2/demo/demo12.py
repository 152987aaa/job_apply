import matplotlib.pyplot as plt
import paddle

# 构造x轴数据
x = paddle.linspace(-5, 5, 200, "float32")
# 创建ELU激活函数层
elu = paddle.nn.ELU()
# 执行计算
y = elu(x)
# 输出打印
plt.plot(x.numpy(), y.numpy(), c="red")
plt.savefig("result.png")
