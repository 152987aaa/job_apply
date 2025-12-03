import paddle

# Tensor的加减乘除运算
x = paddle.to_tensor([[1, 2], [3, 4]])
y = paddle.to_tensor([[5, 6], [7, 8]])
# z = x + y
z = paddle.add(x, y)
y.add_(x)
print(z)
print(y)

# 减法
z2 = paddle.subtract(x, y)
print(z2)
# 乘法
z3 = paddle.multiply(x, y)
print(z3)
# 除法
z4 = paddle.divide(x, y)
print(z4)
