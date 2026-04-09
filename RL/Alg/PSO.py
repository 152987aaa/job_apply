"""粒子群算法(PSO)求解函数优化问题"""
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class PSO:
    def __init__(self, func, dim, pop_size=30, max_iter=100,
                 lb=-10, ub=10, w=0.7, c1=1.5, c2=1.5):
        """
        粒子群算法类

        Parameters:
        -----------
        func : 目标函数
        dim : 搜索空间维度
        pop_size : 粒子数量
        max_iter : 最大迭代次数
        lb, ub : 搜索空间边界
        w : 惯性权重
        c1, c2 : 学习因子
        """
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb, self.ub = lb, ub
        self.w = w
        self.c1, self.c2 = c1, c2

        # 初始化粒子位置和速度
        self.X = np.random.uniform(lb, ub, (pop_size, dim))
        self.V = np.random.uniform(-abs(ub-lb)/10, abs(ub-lb)/10, (pop_size, dim))

        # 初始化个体最优和全局最优
        self.pBest_X = self.X.copy()
        self.pBest_score = np.array([func(x) for x in self.X])
        self.gBest_X = self.pBest_X[np.argmin(self.pBest_score)].copy()
        self.gBest_score = np.min(self.pBest_score)

        # 记录收敛过程
        self.history = [self.gBest_score]

    def update(self):
        """更新粒子位置和速度"""
        r1, r2 = np.random.rand(), np.random.rand()

        # 更新速度
        self.V = (self.w * self.V +
                  self.c1 * r1 * (self.pBest_X - self.X) +
                  self.c2 * r2 * (self.gBest_X - self.X))

        # 速度边界限制
        v_max = abs(self.ub - self.lb) / 2
        self.V = np.clip(self.V, -v_max, v_max)

        # 更新位置
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)

    def optimize(self):
        """执行优化"""
        for _ in range(self.max_iter):
            # 更新粒子
            self.update()

            # 计算适应度
            scores = np.array([self.func(x) for x in self.X])

            # 更新个体最优
            better = scores < self.pBest_score
            self.pBest_X[better] = self.X[better]
            self.pBest_score[better] = scores[better]

            # 更新全局最优
            min_idx = np.argmin(self.pBest_score)
            if self.pBest_score[min_idx] < self.gBest_score:
                self.gBest_X = self.pBest_X[min_idx].copy()
                self.gBest_score = self.pBest_score[min_idx]

            self.history.append(self.gBest_score)

        return self.gBest_X, self.gBest_score


# 目标函数：Sphere函数
def sphere(x):
    return np.sum(x ** 2)


# 运行PSO优化
if __name__ == "__main__":
    pso = PSO(func=sphere, dim=2, pop_size=30, max_iter=100,
              lb=-5, ub=5, w=0.7, c1=1.5, c2=1.5)

    best_pos, best_score = pso.optimize()

    print("=" * 50)
    print("PSO优化结果")
    print("=" * 50)
    print(f"最优位置: {best_pos}")
    print(f"最优适应度: {best_score:.6e}")
    print(f"理论最优: [0, 0], 适应度=0")

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(pso.history, 'b-', linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('全局最优适应度', fontsize=12)
    plt.title('PSO收敛曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('pso_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()