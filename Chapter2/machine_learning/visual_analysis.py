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

# 可视化展示
plt.plot(x_train, y_train, "go", label="Original Data")
plt.xlabel("investment")
plt.ylabel("income")
plt.legend()

# 保存可视化结果
plt.savefig("analysis.png")
