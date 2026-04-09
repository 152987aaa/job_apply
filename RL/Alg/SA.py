"""
模拟退火算法求解 Rastrigin 函数最小值
目标函数: f(x) = x² - 10*cos(2πx) + 10
搜索空间: x ∈ [-5, 5]
全局最优: x* = 0, f(x*) = 0
"""

import numpy as np
import matplotlib.pyplot as plt

# ===================== 中文字体设置 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 算法参数配置 =====================
T0 = 100.0  # 初始温度
T_END = 1e-6  # 终止温度
ALPHA = 0.95  # 降温系数
L = 100  # 每个温度下的迭代次数（链长）
X_MIN = -5.0  # 变量下界
X_MAX = 5.0  # 变量上界
STEP_SIZE = 0.5  # 邻域扰动步长


# ===================== 目标函数 =====================
def objective(x):
    """
    Rastrigin函数：多峰测试函数
    全局最优: x = 0, f(x) = 0
    """
    return x ** 2 - 10 * np.cos(2 * np.pi * x) + 10


# ===================== 邻域生成 =====================
def generate_neighbor(x):
    """
    在当前解的邻域内产生新解
    采用随机扰动法，并约束在可行域内
    """
    # 高斯扰动
    delta = np.random.uniform(-STEP_SIZE, STEP_SIZE)
    x_new = x + delta
    # 边界处理：反射到可行域
    if x_new < X_MIN:
        x_new = 2 * X_MIN - x_new
    if x_new > X_MAX:
        x_new = 2 * X_MAX - x_new
    # 再次检查边界
    x_new = np.clip(x_new, X_MIN, X_MAX)
    return x_new


# ===================== Metropolis准则 =====================
def metropolis(delta_e, temperature):
    """
    Metropolis接受准则
    返回: 是否接受新解 (True/False)
    """
    if delta_e < 0:  # 更优解，必然接受
        return True
    else:  # 劣解，以一定概率接受
        prob = np.exp(-delta_e / temperature)
        return np.random.random() < prob


# ===================== 主算法 =====================
def simulated_annealing():
    """模拟退火算法主流程"""

    # Step 1: 初始化
    x_current = np.random.uniform(X_MIN, X_MAX)  # 随机初始解
    f_current = objective(x_current)

    x_best = x_current  # 最优解
    f_best = f_current  # 最优目标值

    T = T0  # 当前温度

    # 记录搜索过程（用于可视化）
    history_x = [x_current]
    history_f = [f_current]
    history_T = [T]
    history_best_f = [f_best]

    iteration = 0

    print("=" * 50)
    print("模拟退火算法运行中...")
    print("=" * 50)

    # Step 7: 主循环（温度迭代）
    while T > T_END:
        # Step 5-6: 内循环（同一温度下的多次迭代）
        for _ in range(L):
            # Step 2: 产生新解
            x_new = generate_neighbor(x_current)
            f_new = objective(x_new)

            # Step 3: 计算目标函数差
            delta_e = f_new - f_current

            # Step 4: Metropolis准则决定是否接受
            if metropolis(delta_e, T):
                x_current = x_new
                f_current = f_new

            # Step 5: 更新最优解
            if f_current < f_best:
                x_best = x_current
                f_best = f_current

        # Step 6: 降温
        T = ALPHA * T
        iteration += 1

        # 记录历史
        history_x.append(x_current)
        history_f.append(f_current)
        history_T.append(T)
        history_best_f.append(f_best)

        # 输出进度
        if iteration % 10 == 0:
            print(f"迭代{iteration:3d}: T={T:10.6f}, "
                  f"当前x={x_current:7.4f}, f={f_current:7.4f}, "
                  f"最优f={f_best:7.4f}")

    print("=" * 50)
    print(f"最终结果: x = {x_best:.6f}, f(x) = {f_best:.6f}")
    print(f"理论最优: x = 0.0, f(x) = 0.0")
    print(f"总迭代次数: {iteration}")
    print("=" * 50)

    return x_best, f_best, history_x, history_f, history_T, history_best_f


# ===================== 结果可视化 =====================
def plot_results(history_x, history_f, history_T, history_best_f):
    """绘制搜索过程和收敛曲线"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 图1: 目标函数曲面与搜索轨迹
    ax1 = axes[0]
    x_range = np.linspace(X_MIN, X_MAX, 1000)
    y_range = objective(x_range)
    ax1.plot(x_range, y_range, 'b-', linewidth=1.5, label='目标函数')
    ax1.scatter(history_x[::5], [objective(x) for x in history_x[::5]],
                c='red', s=10, alpha=0.3, label='搜索轨迹')
    ax1.scatter([history_x[-1]], [objective(history_x[-1])],
                c='green', s=100, marker='*', zorder=5, label='最终解')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('f(x)', fontsize=11)
    ax1.set_title('目标函数与搜索轨迹', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 图2: 目标函数值收敛曲线
    ax2 = axes[1]
    ax2.plot(history_f, 'r-', linewidth=1.5, alpha=0.7, label='当前解')
    ax2.plot(history_best_f, 'g-', linewidth=2, label='最优解')
    ax2.set_xlabel('迭代次数', fontsize=11)
    ax2.set_ylabel('目标函数值', fontsize=11)
    ax2.set_title('目标函数值收敛曲线', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 图3: 温度下降曲线
    ax3 = axes[2]
    ax3.plot(history_T, 'b-', linewidth=2)
    ax3.set_xlabel('迭代次数', fontsize=11)
    ax3.set_ylabel('温度 T', fontsize=11)
    ax3.set_title('温度下降曲线', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sa_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n可视化结果已保存至: sa_convergence.png")


# ===================== 运行算法 =====================
if __name__ == "__main__":
    # 打印参数配置
    print("=" * 50)
    print("模拟退火算法参数配置")
    print("=" * 50)
    print(f"初始温度: T0 = {T0}")
    print(f"终止温度: T_end = {T_END}")
    print(f"降温系数: α = {ALPHA}")
    print(f"链长: L = {L}")
    print(f"搜索空间: [{X_MIN}, {X_MAX}]")
    print(f"邻域步长: {STEP_SIZE}")
    print("=" * 50)

    # 运行算法
    x_best, f_best, h_x, h_f, h_T, h_best = simulated_annealing()

    # 绘制结果
    plot_results(h_x, h_f, h_T, h_best)
