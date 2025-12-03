import paddle

# 执行矩阵乘法
x = paddle.to_tensor([[1, 2], [3, 4], [5, 6]], dtype="float32")
y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype="float32")
z = paddle.matmul(x, y)
print(z)
