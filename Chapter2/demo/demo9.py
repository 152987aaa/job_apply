# import paddle
# import paddle.nn as nn
#
# # 创建输入
# x = paddle.uniform((2, 3, 8, 8), dtype="float32", min=0.0, max=1.0)
# # 构造卷积层
# conv = nn.Conv2D(3, 6, (5, 5))
# # 执行一层卷积
# y = conv(x)
# # 输出tensor转numpy
# y_np = y.numpy()
# print(y_np.shape)
import paddle
import paddle.nn as nn
# 创建输入
x = paddle.uniform((2,3,8,8),dtype="float32",min=0.0,max=1.0)
# 创建卷积层
conv = nn.Conv2D(3,6,(5,5))
# 执行一层卷积
y = conv(x)
# 将tensor转numpy
y_np = y.numpy()
print(y_np.shape)
