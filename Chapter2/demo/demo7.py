import paddle

# reshape调整Tensor形状
x = paddle.to_tensor([[1, 2], [3, 4], [5, 6]], dtype="float32")
y = paddle.to_tensor([1, 2, 3, 4, 5, 6], dtype="float32")
paddle.reshape_(y, [2, 3])
z = paddle.matmul(x, y)
print(z)
