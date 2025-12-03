import matplotlib.pyplot as plt
import numpy as np

# 输入数据
x_train = np.array(
    [3.3, 4.4, 5.5, 6.7, 6.9, 4.2, 9.8, 6.2, 7.6, 2.2, 7, 10.8, 5.3, 8, 3.1],
    dtype=np.float32,
)
y_train = np.array(
    [17, 28, 21, 32, 17, 16, 34, 26, 25, 12, 28, 35, 17, 29, 13], dtype=np.float32
)

# 求解直线方程的a和b
n = len(x_train)
sum_xy = sum(x_train * y_train)
sum_x = sum(x_train)
sum_y = sum(y_train)
sum_x2 = sum(pow(x_train, 2))
a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 0.00001)
b = (sum_x2 * sum_y - sum_x * sum_xy) / (n * sum_x2 - sum_x * sum_x + 0.00001)

# 可视化输出
x_pred = np.arange(0, 15)
y_pred = a * x_pred + b
plt.plot(x_train, y_train, "go", label="Original Data")
plt.plot(x_pred, y_pred, "r-", label="Fitted Line")
plt.xlabel("investment")
plt.ylabel("income")
plt.legend()
plt.savefig("result.png")

# 预测第16年的收益值
x = 12.5
y = a * x + b
print(y)
