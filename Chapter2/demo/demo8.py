import paddle
import numpy as np

x = np.ones([2, 2]) * 3
x = paddle.to_tensor(x, stop_gradient=False)
y = paddle.pow(x, 2)
y.backward()
print((x.grad).numpy())
