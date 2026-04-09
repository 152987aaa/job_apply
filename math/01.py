import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ========== 1. 设置绘图环境 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
fig, ax = plt.subplots(figsize=(10, 8))

# ========== 2. 定义变量范围 ==========
# x2 (纵坐标) 的取值范围，范围要大一点以覆盖所有约束
x2 = np.linspace(-0.5, 6, 500)

# ========== 3. 绘制约束边界 ==========
# 约束1：抛物线边界 x1 = 5*x2 - x2^2
x1_parabola = 5 * x2 - x2**2
ax.plot(x1_parabola, x2, 'b-', linewidth=2, label=r'约束曲线: $x_1 + x_2^2 - 5x_2 = 0$')

# 约束2：直线边界 x1 = 5 - x2
x1_line = 5 - x2
ax.plot(x1_line, x2, 'r-', linewidth=2, label=r'约束直线: $x_1 + x_2 - 5 = 0$')

# 约束3：非负轴
x1_0 = np.zeros_like(x2)
x2_0 = np.zeros_like(x1_parabola)
ax.plot(x1_0, x2, 'k-', linewidth=1.5, label=r'$x_1=0$')
ax.plot(x1_parabola, x2_0, 'k-', linewidth=1.5, label=r'$x_2=0$')

# ========== 4. 绘制可行域 (Fill Between) ==========
# 可行域条件：
# 1. x1 >= 抛物线 (x1 >= 5*x2 - x2**2)
# 2. x1 >= 直线 (x1 >= 5 - x2)
# 3. x1 >= 0, x2 >= 0
# 取上界的最小值，即同时满足 x1 >= max(抛物线, 直线, 0)
x1_upper_bound = np.maximum(x1_parabola, x1_line)
x1_upper_bound = np.maximum(x1_upper_bound, 0) # 保证非负

# 填充可行域（在抛物线和直线之上的区域）
# 我们将y轴上限设为6，下限设为x1_upper_bound
ax.fill_betweenx(x2, x1_upper_bound, 6,
                 where=(x1_upper_bound < 6),
                 color='lightgreen', alpha=0.4, label='可行域 Feasible Region')

# ========== 5. 绘制目标函数等值线 (等高线) ==========
# 目标函数：(x1-2)^2 + (x2-1)^2 = f
# 画几个不同半径的圆，展示最小化过程
centers = np.array([[2, 1]]) # 目标函数圆心 (2,1)
radii = [0.5, 1.0, 1.5, 2.5, 3.5] # 不同的半径，展示逐渐逼近

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(radii)))
for i, r in enumerate(radii):
    circle = plt.Circle(centers[0], r, fill=False, color=colors[i],
                        linestyle='--', linewidth=1.5, label=f'等值线 r={r}')
    ax.add_patch(circle)

# ========== 6. 计算并标记最优点 (数值解法辅助) ==========
# 定义目标函数
def func(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

# 定义约束条件 (scipy格式)
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] + x[1]**2 - 5 * x[1]}, # x1 + x2^2 -5x2 >= 0
        {'type': 'ineq', 'fun': lambda x:  x[0] + x[1] - 5})         # x1 + x2 -5 >= 0

# 初始猜测值 (x1, x2)
x0 = [2, 2]

# 求解优化问题
# bounds: x1>=0, x2>=0
bnds = ((0, None), (0, None))
result = minimize(func, x0, method='SLSQP', bounds=bnds, constraints=cons)

opt_x1, opt_x2 = result.x[0], result.x[1]
opt_val = result.fun

# 绘制最优点
ax.plot(opt_x1, opt_x2, 'ko', markersize=8, label=f'最优点 ({opt_x1:.2f}, {opt_x2:.2f})')
# 绘制从圆心指向最优点的连线（展示距离）
ax.plot([2, opt_x1], [1, opt_x2], 'g--', alpha=0.6, linewidth=1, label='最短距离向量')

# ========== 7. 图形美化与显示 ==========
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_title('图解法求解非线性规划：例3', fontsize=16, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.6)

# 标记圆心
ax.plot(2, 1, 'rs', markersize=5, label='目标函数圆心 (2,1)')

plt.tight_layout()
plt.show()

# 输出具体的最优解数值
print(f"计算得出的最优解 (x1, x2): ({opt_x1:.4f}, {opt_x2:.4f})")
print(f"最优目标函数值: {opt_val:.4f}")