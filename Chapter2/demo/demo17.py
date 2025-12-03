import paddle
import numpy as np

# 构造输入数据
inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
inp = paddle.to_tensor(inp)
# 构造输入数据的真值标签
label = paddle.to_tensor(1.0)
# 创建线性变换层
linear = paddle.nn.Linear(10, 10)
# 定义损失函数
mse_loss = paddle.nn.MSELoss()
# 定义优化器
beta1 = paddle.to_tensor([0.9], dtype="float32")
beta2 = paddle.to_tensor([0.99], dtype="float32")
adam = paddle.optimizer.Adam(
    learning_rate=0.1,
    parameters=linear.parameters(),
    beta1=beta1,
    beta2=beta2,
    weight_decay=0.01,
)
# 前向计算
out = paddle.mean(linear(inp))
# 计算迭代前损失函数
loss = mse_loss(out, label)
print("迭代前损失函数:", float(loss))
# 后向传播，更新参数
loss.backward()
adam.step()
adam.clear_grad()
# 计算迭代后损失函数
out = paddle.mean(linear(inp))
loss = mse_loss(out, label)
print("迭代后损失函数:", float(loss))
