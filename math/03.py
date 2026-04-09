import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 创建画布
plt.figure(figsize=(10, 5))

# -------- 左图：凸集（实心圆） --------
plt.subplot(1, 2, 1)
# 画圆
theta = np.linspace(0, 2*np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
plt.fill(x, y, color="lightblue", alpha=0.5, label="凸集")
# 随便连两个点，线段都在内部
plt.plot([0.5, -0.6], [0.5, 0.6], "r-", linewidth=2, label="内部连线")
plt.title("凸集（实心圆）\n任意两点连线都在内部", fontsize=12)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.legend()

# -------- 右图：非凸集（月牙形） --------
plt.subplot(1, 2, 2)
# 画月牙（非凸）
theta1 = np.linspace(-np.pi/2, np.pi/2, 100)
x1 = np.cos(theta1)
y1 = np.sin(theta1)
x2 = 0.7*np.cos(theta1+np.pi)
y2 = 0.7*np.sin(theta1+np.pi)
x = np.concatenate([x1, x2[::-1]])
y = np.concatenate([y1, y2[::-1]])
plt.fill(x, y, color="lightcoral", alpha=0.5, label="非凸集")
# 两点连线跑到外面了！
plt.plot([0.8, 0.8], [0.6, -0.6], "r-", linewidth=2, label="连线穿出集合")
plt.title("非凸集（月牙）\n两点连线跑到外面", fontsize=12)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()