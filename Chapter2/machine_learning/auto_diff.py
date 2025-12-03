import matplotlib.pyplot as plt
import numpy as np
import paddle

# 输入数据
x_train = np.array(
    [3.3, 4.4, 5.5, 6.7, 6.9, 4.2, 9.8, 6.2, 7.6, 2.2, 7, 10.8, 5.3, 8, 3.1],
    dtype=np.float32,
)
y_train = np.array(
    [17, 28, 21, 32, 17, 16, 34, 26, 25, 12, 28, 35, 17, 29, 13], dtype=np.float32
)

# numpy转tensor
x_train = paddle.to_tensor(x_train)
y_train = paddle.to_tensor(y_train)

# 随机初始化模型参数
a = np.random.randn(1)
a = paddle.to_tensor(a, dtype="float32", stop_gradient=False)
b = np.random.randn(1)
b = paddle.to_tensor(b, dtype="float32", stop_gradient=False)

# 循环迭代
for t in range(10):
    # 计算平方差损失
    y_ = a * x_train + b
    loss = paddle.sum((y_ - y_train) ** 2)
    # 自动计算梯度
    loss.backward()
    # 更新参数（梯度下降），学习率默认使用1e-3
    a = a.detach() - 1e-3 * float(a.grad)
    b = b.detach() - 1e-3 * float(b.grad)
    a.stop_gradient = False
    b.stop_gradient = False
    # 输出当前轮的目标函数值L
    print("epoch: {}, loss: {}".format(t, (float(loss))))

# 训练结束，终止a和b的梯度计算
a.stop_gradient = True
b.stop_gradient = True

# 可视化输出
x_pred = paddle.arange(0, 15)
y_pred = a * x_pred + b
plt.plot(x_train.numpy(), y_train.numpy(), "go", label="Original Data")
plt.plot(x_pred.numpy(), y_pred.numpy(), "r-", label="Fitted Line")
plt.xlabel("investment")
plt.ylabel("income")
plt.legend()
plt.savefig("result.png")

# 预测第16年的收益值
x = 12.5
y = a * x + b
print(y.numpy())