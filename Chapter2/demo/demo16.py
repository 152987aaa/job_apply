import paddle
# 创建对比数据
pred = paddle.to_tensor(1.5)
label = paddle.to_tensor(1.7)
# 定义损失函数
mse_loss = paddle.nn.MSELoss()
# 计算损失函数
output = mse_loss(pred, label)
print(output.numpy())