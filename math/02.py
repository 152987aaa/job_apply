import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# ========== 1. 设置绘图环境 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


from mpl_toolkits.mplot3d import Axes3D


# -------------------------- 1. 定义目标函数与梯度 --------------------------
def f(x1, x2):
    """题目中的非凸函数"""
    u = 1 - 0.5 * x1 + x1 ** 5 + x2 ** 3
    v = np.exp(-(x1 ** 2 + x2 ** 2))
    return u * v


def grad_f(x1, x2):
    """计算函数在(x1, x2)处的梯度（偏导数）"""
    u = 1 - 0.5 * x1 + x1 ** 5 + x2 ** 3
    v = np.exp(-(x1 ** 2 + x2 ** 2))

    # 对x1的偏导数
    df_dx1 = v * (-0.5 + 5 * x1 ** 4 - 2 * x1 * u)
    # 对x2的偏导数
    df_dx2 = v * (3 * x2 ** 2 - 2 * x2 * u)

    return np.array([df_dx1, df_dx2])


# -------------------------- 2. 生成网格数据 --------------------------
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
F = f(X1, X2)

# -------------------------- 3. 创建3D绘图 --------------------------
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D曲面（降低透明度，让梯度更显眼）
surf = ax.plot_surface(
    X1, X2, F,
    cmap='viridis',
    alpha=0.5,  # 降低透明度
    edgecolor='lightgray',  # 加网格线，更立体
    antialiased=True
)

# 绘制底部的等值线（等高线）
ax.contour(
    X1, X2, F,
    zdir='z',
    offset=np.min(F) - 0.3,
    cmap='viridis',
    linewidths=2,
    levels=10
)

# -------------------------- 4. 标记指定点并绘制梯度向量（优化版） --------------------------
points = np.array([
    [0, 1.8],  # 点1
    [-0.9, -1],  # 点2
    [-2.05, 0.2]  # 点3
])
colors = ['red', 'blue', 'green']
labels = [
    '点(0, 1.80) 梯度',
    '点(-0.90, -1.00) 梯度',
    '点(-2.05, 0.20) 梯度'
]

for i, (x1_p, x2_p) in enumerate(points):
    f_p = f(x1_p, x2_p)
    grad_p = grad_f(x1_p, x2_p)

    # 关键优化：
    # 1. 增大scale让箭头更长
    # 2. 抬高z轴起点(f_p + 0.1)，避免被曲面挡住
    # 3. 加粗线宽，用醒目颜色
    scale = 1.0  # 放大梯度长度
    ax.quiver(
        x1_p, x2_p, f_p + 0.1,  # 箭头起点：在点上方一点
                    grad_p[0] * scale, grad_p[1] * scale, 0,  # 箭头方向
        color=colors[i],
        linewidth=3,
        arrow_length_ratio=0.3,  # 箭头头部比例
        label=labels[i]
    )
    # 标记该点（用黑色实心圆，更醒目）
    ax.scatter(x1_p, x2_p, f_p, color='black', s=100, marker='o', edgecolor='white', linewidth=2)

# -------------------------- 5. 美化图表与视角 --------------------------
ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_zlabel('$f(x_1, x_2)$', fontsize=14)
ax.set_title(
    '非凸函数 $f(x_1,x_2) = (1 - \\frac{1}{2}x_1 + x_1^5 + x_2^3)e^{-(x_1^2+x_2^2)}$\n梯度与等值线垂直关系（优化版）',
    fontsize=16, pad=20
)
ax.legend(loc='upper right', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=10, label='$f(x_1, x_2)$ 数值')

# 调整视角：让梯度箭头完全暴露在视野中
ax.view_init(elev=40, azim=-70)  # 抬高视角，向左旋转

plt.tight_layout()
plt.show()