import paddle
import numpy as np
# Numpy 转 Tensor
x = np.ones([2, 3])
y = paddle.to_tensor(x)
# Tensor 转 Numpy
z = y.numpy()
print(z)