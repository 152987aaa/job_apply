import paddle
import numpy as np

x = np.ones([2, 3])
y = paddle.to_tensor(x)
print(y)
