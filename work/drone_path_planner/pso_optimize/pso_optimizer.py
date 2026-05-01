"""
浙大高飞无人机团队 | PSO路径优化器
【严格对应链接步骤2】
功能：以A*输出的路点为初始种群 → 优化路径长度 → 输出优化后路点
教学重点：粒子群优化原理、惯性权重、认知/社会系数
"""

import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import load_waypoints, save_waypoints, calculate_path_length, plot_pso_convergence

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== PSO参数（可调参学习）=====================
NUM_PARTICLES = 30       # 粒子数量
MAX_ITER = 100           # 最大迭代次数
W_INERTIA = 0.5          # 惯性权重（控制历史速度影响）
C_COGNITIVE = 1.5        # 认知系数（个人经验权重）
C_SOCIAL = 1.5           # 社会系数（群体经验权重）
MAX_VELOCITY = 2.0       # 最大速度限制

# ===================== 粒子类 =====================
class Particle:
    def __init__(self, waypoints, bounds):
        """
        初始化单个粒子
        :param waypoints: A*输出的原始路点
        :param bounds: 坐标边界 [(min_x, max_x), (min_y, max_y)]
        """
        # 提取起点和终点（固定不变）
        self.start = waypoints[0]
        self.end = waypoints[-1]
        # 中间路点数量
        self.num_middle = len(waypoints) - 2
        
        # 初始化粒子位置（在原始路点附近添加随机扰动）
        self.position = []
        for i in range(1, len(waypoints) - 1):
            x = waypoints[i][0] + np.random.uniform(-2, 2)
            y = waypoints[i][1] + np.random.uniform(-2, 2)
            # 限制在边界内
            x = max(bounds[0][0], min(bounds[0][1], x))
            y = max(bounds[1][0], min(bounds[1][1], y))
            self.position.append([x, y])
        self.position = np.array(self.position)
        
        # 初始化速度（随机小值）
        self.velocity = np.random.uniform(-0.5, 0.5, self.position.shape)
        
        # 初始化个人最优
        self.pbest_position = np.copy(self.position)
        self.pbest_fitness = self.calculate_fitness()
    
    def calculate_fitness(self):
        """
        计算适应度：路径总长度（越小越好）
        :return: 路径长度
        """
        total_length = 0.0
        
        # 起点到第一个中间点
        total_length += np.hypot(self.start[0] - self.position[0][0],
                                self.start[1] - self.position[0][1])
        
        # 中间点之间
        for i in range(len(self.position) - 1):
            total_length += np.hypot(self.position[i+1][0] - self.position[i][0],
                                    self.position[i+1][1] - self.position[i][1])
        
        # 最后一个中间点到终点
        total_length += np.hypot(self.end[0] - self.position[-1][0],
                                self.end[1] - self.position[-1][1])
        
        return total_length
    
    def update_velocity(self, gbest_position):
        """
        更新粒子速度（PSO核心公式）
        :param gbest_position: 全局最优位置
        """
        # 随机因子
        r1 = np.random.random(self.position.shape)
        r2 = np.random.random(self.position.shape)
        
        # PSO速度更新公式：
        # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        cognitive = C_COGNITIVE * r1 * (self.pbest_position - self.position)
        social = C_SOCIAL * r2 * (gbest_position - self.position)
        self.velocity = W_INERTIA * self.velocity + cognitive + social
        
        # 限制速度范围
        self.velocity = np.clip(self.velocity, -MAX_VELOCITY, MAX_VELOCITY)
    
    def update_position(self, bounds):
        """
        更新粒子位置
        :param bounds: 坐标边界
        """
        self.position += self.velocity
        
        # 限制位置在边界内
        for i in range(len(self.position)):
            self.position[i][0] = max(bounds[0][0], min(bounds[0][1], self.position[i][0]))
            self.position[i][1] = max(bounds[1][0], min(bounds[1][1], self.position[i][1]))
        
        # 更新个人最优
        current_fitness = self.calculate_fitness()
        if current_fitness < self.pbest_fitness:
            self.pbest_position = np.copy(self.position)
            self.pbest_fitness = current_fitness
    
    def get_full_path(self):
        """获取完整路径（包含起点和终点）"""
        path = [self.start]
        for point in self.position:
            path.append(tuple(point))
        path.append(self.end)
        return path

# ===================== PSO优化器类 =====================
class PSOOptimizer:
    def __init__(self, waypoints, bounds):
        """
        初始化PSO优化器
        :param waypoints: A*输出的路点
        :param bounds: 坐标边界
        """
        self.waypoints = waypoints
        self.bounds = bounds
        
        # 初始化粒子群
        self.particles = []
        for _ in range(NUM_PARTICLES):
            particle = Particle(waypoints, bounds)
            self.particles.append(particle)
        
        # 初始化全局最优
        self.gbest_position = None
        self.gbest_fitness = float('inf')
        self.update_gbest()
    
    def update_gbest(self):
        """更新全局最优位置"""
        for particle in self.particles:
            if particle.pbest_fitness < self.gbest_fitness:
                self.gbest_fitness = particle.pbest_fitness
                self.gbest_position = np.copy(particle.pbest_position)
    
    def optimize(self):
        """
        执行PSO优化
        :return: (最优路径, 适应度历史)
        """
        fitness_history = []
        
        print(f"开始PSO优化")
        print(f"粒子数量: {NUM_PARTICLES}")
        print(f"最大迭代: {MAX_ITER}")
        print(f"初始最优路径长度: {self.gbest_fitness:.2f}")
        
        for iteration in range(MAX_ITER):
            # 更新每个粒子
            for particle in self.particles:
                particle.update_velocity(self.gbest_position)
                particle.update_position(self.bounds)
            
            # 更新全局最优
            self.update_gbest()
            
            # 记录历史
            fitness_history.append(self.gbest_fitness)
            
            # 打印进度
            if (iteration + 1) % 20 == 0:
                print(f"迭代 {iteration+1}/{MAX_ITER}: 最优路径长度 = {self.gbest_fitness:.2f}")
        
        print(f"\nPSO优化完成！")
        print(f"最终最优路径长度: {self.gbest_fitness:.2f}")
        
        # 返回最优路径
        best_path = None
        for particle in self.particles:
            if particle.pbest_fitness == self.gbest_fitness:
                best_path = particle.get_full_path()
                break
        
        return best_path, fitness_history

# ===================== 主函数 =====================
def main():
    print("=" * 60)
    print("PSO路径优化器")
    print("浙大高飞无人机团队")
    print("=" * 60)
    
    # 步骤1：加载A*输出的路点
    print("\n【步骤1】加载A*路点")
    try:
        waypoints = load_waypoints("../data/output/a_star_waypoints.txt")
        print(f"Loaded {len(waypoints)} waypoints successfully")
    except Exception as e:
        print(f"Failed to load A* waypoints: {e}")
        print("Please run A* planner first")
        return
    
    # 步骤2：设置优化参数
    print("\n【步骤2】设置优化参数")
    bounds = [(0, 25), (0, 25)]  # 地图边界
    print(f"坐标边界: {bounds}")
    print(f"惯性权重 w: {W_INERTIA}")
    print(f"认知系数 c1: {C_COGNITIVE}")
    print(f"社会系数 c2: {C_SOCIAL}")
    
    # 步骤3：执行PSO优化
    print("\n【步骤3】执行PSO优化")
    optimizer = PSOOptimizer(waypoints, bounds)
    optimized_path, fitness_history = optimizer.optimize()
    
    # 步骤4：保存结果
    print("\n【步骤4】保存优化结果")
    path_array = np.array(optimized_path)
    save_waypoints("../data/output/pso_optimized_waypoints.txt", path_array)
    
    # 计算优化前后路径长度对比
    original_length = calculate_path_length(waypoints)
    optimized_length = calculate_path_length(optimized_path)
    improvement = ((original_length - optimized_length) / original_length) * 100
    print(f"原始路径长度: {original_length:.2f} 米")
    print(f"优化后路径长度: {optimized_length:.2f} 米")
    print(f"路径缩短: {improvement:.2f}%")
    
    # 可视化1：收敛曲线
    plot_pso_convergence(fitness_history, "PSO收敛曲线", "../data/output/pso_convergence.png")
    
    # 可视化2：优化前后路径对比图
    plt.figure(figsize=(10, 8))
    plt.plot([p[0] for p in waypoints], [p[1] for p in waypoints], 'b--', linewidth=2, label=f'原始路径 ({original_length:.2f}米)')
    plt.plot([p[0] for p in optimized_path], [p[1] for p in optimized_path], 'r-', linewidth=2.5, label=f'优化路径 ({optimized_length:.2f}米)')
    plt.scatter(waypoints[0][0], waypoints[0][1], c='green', s=150, marker='o', label='起点')
    plt.scatter(waypoints[-1][0], waypoints[-1][1], c='blue', s=150, marker='*', label='终点')
    plt.title('PSO路径优化对比', fontsize=14)
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('../data/output/pso_path_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("路径对比图已保存")
    
    # 可视化3：优化前后长度柱状对比
    plt.figure(figsize=(6, 5))
    plt.bar(['原始路径', '优化路径'], [original_length, optimized_length], color=['blue', 'red'])
    plt.title('路径长度对比', fontsize=14)
    plt.ylabel('长度 (米)', fontsize=12)
    plt.text(0, original_length + 0.5, f'{original_length:.2f}', ha='center', fontsize=12)
    plt.text(1, optimized_length + 0.5, f'{optimized_length:.2f}', ha='center', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('../data/output/pso_length_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("长度对比柱状图已保存")
    
    # 可视化4：PSO粒子群动画（GIF）
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([p[0] for p in waypoints], [p[1] for p in waypoints], 'b--', linewidth=2, label='原始路径')
    global_best_line, = ax.plot([], [], 'r-', linewidth=2.5, label='当前最优')
    particles_scatter = ax.scatter([], [], c='orange', s=50, label='粒子群')
    ax.scatter(waypoints[0][0], waypoints[0][1], c='green', s=150, marker='o', label='起点')
    ax.scatter(waypoints[-1][0], waypoints[-1][1], c='blue', s=150, marker='*', label='终点')
    ax.set_title('PSO粒子群优化过程', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 26)
    ax.set_ylim(-1, 26)
    
    # 重新运行一次获取粒子位置历史
    optimizer2 = PSOOptimizer(waypoints, bounds)
    particle_history = []
    
    for _ in range(MAX_ITER):
        positions = []
        for particle in optimizer2.particles:
            positions.append(particle.position.copy())
            particle.update_velocity(optimizer2.gbest_position)
            particle.update_position(optimizer2.bounds)
        optimizer2.update_gbest()
        particle_history.append((np.array(positions), optimizer2.gbest_position.copy()))
    
    def animate_pso(i):
        positions, gbest = particle_history[i]
        particles_scatter.set_offsets(positions.reshape(-1, 2))
        global_best_line.set_data(gbest[:, 0], gbest[:, 1])
        return particles_scatter, global_best_line
    
    anim = animation.FuncAnimation(fig, animate_pso, frames=min(50, MAX_ITER), interval=150, blit=True)
    anim.save('../data/output/pso_animation.gif', writer='pillow', dpi=100)
    plt.close()
    print("PSO优化动画GIF已保存")
    
    print("\n[PSO Done] Path optimization completed!")
    print("Output files:")
    print("  - data/output/pso_optimized_waypoints.txt: 优化路点（供B-spline使用）")
    print("  - data/output/pso_convergence.png: 收敛曲线")
    print("  - data/output/pso_path_comparison.png: 路径对比图")
    print("  - data/output/pso_length_comparison.png: 长度对比柱状图")
    print("  - data/output/pso_animation.gif: 粒子群优化动画")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()