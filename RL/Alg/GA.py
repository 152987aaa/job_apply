"""
遗传算法求解 f(x) = x² 在 [0,31] 上的最大值
编码方式：5位二进制编码（可表示0-31共32个整数）
"""

import numpy as np
import random
import matplotlib.pyplot as plt

# ===================== 中文字体设置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===================== 算法参数配置 =====================
POP_SIZE = 50  # 种群规模
CHROM_LEN = 10  # 染色体长度（二进制位数，决定精度）
PC = 0.8  # 交叉概率
PM = 0.02  # 变异概率
MAX_GEN = 100  # 最大进化代数
ELITE_SIZE = 1  # 精英个体数量

# ===================== 变量范围配置 =====================
X_MIN = 0.0  # 变量下界
X_MAX = 31.0  # 变量上界


# 编码精度 = (X_MAX - X_MIN) / (2^CHROM_LEN - 1)
# 当 CHROM_LEN=10 时，精度 ≈ 0.0303


# ===================== 编码与解码 =====================
def encode(x):
    """
    将实数值编码为二进制染色体
    x: 实际变量值
    返回: 二进制染色体数组
    """
    # 将 x 映射到 [0, 2^n-1] 的整数
    decimal = round((x - X_MIN) / (X_MAX - X_MIN) * (2 ** CHROM_LEN - 1))
    decimal = max(0, min(decimal, 2 ** CHROM_LEN - 1))  # 边界约束

    # 转换为二进制数组
    chromosome = np.zeros(CHROM_LEN, dtype=int)
    for i in range(CHROM_LEN):
        chromosome[i] = (decimal >> (CHROM_LEN - 1 - i)) & 1
    return chromosome


def decode(chromosome):
    """
    将二进制染色体解码为实数值
    染色体 → 十进制整数 → 映射到 [X_MIN, X_MAX]
    """
    # 二进制转十进制
    decimal = sum(chromosome[i] * (2 ** (CHROM_LEN - 1 - i))
                  for i in range(CHROM_LEN))
    # 映射到实际变量范围
    x = X_MIN + decimal * (X_MAX - X_MIN) / (2 ** CHROM_LEN - 1)
    return x


# ===================== 适应度函数 =====================
def fitness(chromosome):
    """
    计算染色体的适应度值
    解码 → 计算目标函数值
    """
    x = decode(chromosome)
    return x ** 2  # 目标函数 f(x) = x²


# ===================== 遗传操作 =====================
def selection(pop, fit_vals):
    """
    轮盘赌选择：适应度越高，被选中概率越大
    """
    # 计算选择概率
    total = np.sum(fit_vals)
    probs = fit_vals / total if total > 0 else np.ones(POP_SIZE) / POP_SIZE

    # 有放回抽样，选出POP_SIZE个个体
    indices = np.random.choice(POP_SIZE, size=POP_SIZE, p=probs)
    return pop[indices].copy()


def crossover(pop):
    """
    单点交叉：随机位置切断，交换后半段
    """
    new_pop = pop.copy()
    for i in range(0, POP_SIZE - 1, 2):
        if random.random() < PC:
            # 随机选择交叉点（1到CHROM_LEN-1）
            point = random.randint(1, CHROM_LEN - 1)
            # 交换两个个体在交叉点之后的基因
            new_pop[i, point:], new_pop[i + 1, point:] = \
                new_pop[i + 1, point:].copy(), new_pop[i, point:].copy()
    return new_pop


def mutation(pop):
    """
    位翻转变异：每个基因位以概率PM取反
    """
    new_pop = pop.copy()
    for i in range(POP_SIZE):
        for j in range(CHROM_LEN):
            if random.random() < PM:
                new_pop[i, j] = 1 - new_pop[i, j]  # 0↔1
    return new_pop


def elitism(pop, fit_vals, elite_individual, elite_fitness):
    """
    精英策略：将上一代最优个体替换当前代最差个体
    确保最优适应度单调不减

    参数:
        pop: 当前种群
        fit_vals: 当前适应度值
        elite_individual: 上一代最优个体
        elite_fitness: 上一代最优适应度

    返回:
        更新后的种群和适应度值
    """
    # 找到当前代最差个体的索引
    worst_idx = np.argmin(fit_vals)

    # 用上一代最优个体替换当前代最差个体
    pop[worst_idx] = elite_individual.copy()
    fit_vals[worst_idx] = elite_fitness

    return pop, fit_vals


# ===================== 主算法 =====================
def genetic_algorithm():
    """遗传算法主流程（含精英策略）"""

    # Step 1: 初始化种群
    pop = np.random.randint(0, 2, (POP_SIZE, CHROM_LEN))

    # 记录每代的适应度值（用于绘制收敛曲线）
    best_fitness_history = []  # 每代最优适应度
    avg_fitness_history = []  # 每代平均适应度

    # 初始化精英个体
    elite_individual = None
    elite_fitness = -1

    for gen in range(MAX_GEN):
        # Step 2: 适应度评估
        fit_vals = np.array([fitness(ind) for ind in pop])

        # Step 6: 精英策略（从第2代开始）
        if gen > 0 and elite_individual is not None:
            pop, fit_vals = elitism(pop, fit_vals, elite_individual, elite_fitness)

        # 更新精英个体（保存当前代最优）
        max_idx = np.argmax(fit_vals)
        current_best_fit = fit_vals[max_idx]

        # 精英策略核心：只有更优时才更新精英
        if current_best_fit > elite_fitness:
            elite_individual = pop[max_idx].copy()
            elite_fitness = current_best_fit

        # 记录当前代的最优和平均适应度
        best_fitness_history.append(elite_fitness)  # 使用精英适应度，保证单调不减
        avg_fitness_history.append(np.mean(fit_vals))

        # 输出每10代的进化状态
        if gen % 10 == 0:
            print(f"第{gen:3d}代: 当前最优 x={decode(elite_individual):6.2f}, "
                  f"f(x)={elite_fitness:8.2f}")

        # Step 3-5: 选择 → 交叉 → 变异
        pop = selection(pop, fit_vals)
        pop = crossover(pop)
        pop = mutation(pop)

    return decode(elite_individual), elite_fitness, best_fitness_history, avg_fitness_history


# ===================== 绘制收敛曲线 =====================
def plot_convergence(best_history, avg_history):
    """
    绘制适应度收敛曲线（中文版）
    """
    plt.figure(figsize=(10, 6))

    generations = range(len(best_history))

    # 绘制平均适应度曲线
    plt.plot(generations, avg_history,
             color='#f59e0b', linewidth=2,
             label='平均适应度', alpha=0.8)

    # 绘制最优适应度曲线（精英策略保证单调不减）
    plt.plot(generations, best_history,
             color='#3b82f6', linewidth=2.5,
             label='最优适应度（精英）')

    # 标记最优值点
    best_val = best_history[-1]  # 最终值即为最优
    best_gen = len(best_history) - 1
    plt.scatter([best_gen], [best_val],
                color='#3b82f6', s=100, zorder=5,
                edgecolors='white', linewidths=2)
    plt.annotate(f'最优解: {best_val:.2f}',
                 xy=(best_gen, best_val),
                 xytext=(best_gen - 20, best_val + 30),
                 fontsize=11, color='#1e40af',
                 arrowprops=dict(arrowstyle='->', color='#3b82f6'))

    # 图表装饰
    plt.xlabel('进化代数', fontsize=12)
    plt.ylabel('适应度', fontsize=12)
    plt.title('遗传算法收敛曲线（含精英策略）', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')

    # 设置y轴范围
    plt.ylim(0, 1050)

    plt.tight_layout()
    plt.savefig('convergence_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n收敛曲线已保存至: convergence_curve.png")


# ===================== 运行算法 =====================
if __name__ == "__main__":
    # 打印编码精度信息
    precision = (X_MAX - X_MIN) / (2 ** CHROM_LEN - 1)
    print("=" * 50)
    print("遗传算法参数配置")
    print("=" * 50)
    print(f"变量范围: [{X_MIN}, {X_MAX}]")
    print(f"染色体长度: {CHROM_LEN} 位")
    print(f"编码精度: {precision:.6f}")
    print(f"精英策略: 开启 (保留 {ELITE_SIZE} 个精英)")
    print("=" * 50)
    print("遗传算法运行中...")
    print("=" * 50)

    x, f_val, best_hist, avg_hist = genetic_algorithm()

    print("=" * 50)
    print(f"最终结果: x = {x:.4f}, f(x) = {f_val:.4f}")
    print(f"理论最优: x = {X_MAX}, f(x) = {X_MAX ** 2:.0f}")
    print(f"误差: Δx = {abs(x - X_MAX):.6f}")
    print("=" * 50)

    # 绘制收敛曲线
    plot_convergence(best_hist, avg_hist)
