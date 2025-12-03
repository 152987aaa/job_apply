import paddle

# 求取绝对值
x = paddle.to_tensor([[-1, -2], [-3, 4]], dtype="float32")
z1 = paddle.abs(x)
print(z1)

# 求幂操作
z2 = paddle.pow(x,3)
print(z2)