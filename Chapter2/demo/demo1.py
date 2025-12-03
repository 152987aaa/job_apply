import paddle

# 创建5行3列、元素值均为0的Tensor
tensor_empty = paddle.empty(shape=[5, 3], dtype="float32")
print(tensor_empty)

# 创建2行3列、元素值均为1的Tensor
tensor_ones = paddle.ones([2, 3])
print(tensor_ones)

# 创建2行3列、元素值均为0的Tensor
tensor_zeros = paddle.zeros([2, 3])
print(tensor_zeros)

# 创建5行3列、元素值符合[0,1)随机均匀分布的Tensor
tensor_rand = paddle.rand([5, 3])
print(tensor_rand)

# 创建5行3列、元素值符合均值为0且方差为1正态分布的Tensor
tensor_randn = paddle.randn([5, 3])
print(tensor_randn)

# 使用Python的List列表数据来初始化Tensor
tensor_x = paddle.to_tensor([5.5, 4, 8.7, 9])
print(tensor_x)
