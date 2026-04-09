import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 生成网格
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

# 创建画布
fig = plt.figure(figsize=(12, 5))

# -------- 左图：凸函数（碗状，开口向上） --------
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
Z1 = X**2 + Y**2   # 凸函数：z = x² + y²
ax1.plot_surface(X, Y, Z1, cmap="Blues", alpha=0.8)
ax1.set_title("凸函数\n形状=碗，只有1个最低点（全局最优）", fontsize=12)
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# -------- 右图：非凸函数（波浪形，多个坑） --------
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
Z2 = np.sin(X) + np.cos(Y)  # 非凸函数：波浪面
ax2.plot_surface(X, Y, Z2, cmap="Reds", alpha=0.8)
ax2.set_title("非凸函数\n形状=波浪，有很多小坑（局部最优）", fontsize=12)
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.tight_layout()
plt.show()