#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSO路径优化 - 浙江大学高飞无人机团队教学材料

本文件实现了粒子群优化(Particle Swarm Optimization, PSO)算法，
用于优化A*算法生成的路径点，减少路径总长度。

教学目标：
1. 理解PSO算法的基本原理
2. 掌握如何将路径优化问题转化为PSO的优化目标
3. 理解粒子速度和位置的更新规则
4. 学会调整PSO的参数
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import euclidean

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 第一步：回顾A*路径（模拟生成）
# ==============================================================================

def generate_astar_path():
    """
    生成模拟的A*路径点（实际应用中由A*算法生成）
    
    返回：
        waypoints: 路径点列表，每个点为(x, y)元组
    """
    # 模拟一条有优化空间的路径
    # 路径从左下角(0,0)到右上角(20,15)，中间有几个拐点
    waypoints = [
        (0, 0),
        (5, 2),
        (8, 5),
        (12, 6),
        (15, 10),
        (18, 12),
        (20, 15)
    ]
    return waypoints

# ==============================================================================
# 第二步：定义PSO算法
# ==============================================================================

class Particle:
    """
    PSO粒子类
    
    每个粒子代表一条候选路径，存储：
    - 当前位置（路径点坐标）
    - 速度（位置变化量）
    - 个人最优位置（历史最佳路径）
    - 适应度值（路径长度）
    """
    
    def __init__(self, waypoints, bounds):
        """
        初始化粒子
        
        参数：
            waypoints: 原始路径点（用于确定维度）
            bounds: 坐标边界，格式为[(x_min, x_max), (y_min, y_max), ...]
        """
        # 提取中间路径点（不包括起点和终点）
        self.num_middle_points = len(waypoints) - 2
        self.start = waypoints[0]
        self.end = waypoints[-1]
        
        # 初始化粒子位置：在原始路径点附近添加随机扰动
        self.position = []
        for i in range(1, len(waypoints)-1):
            x = waypoints[i][0] + np.random.uniform(-2, 2)
            y = waypoints[i][1] + np.random.uniform(-2, 2)
            # 限制在边界内
            x = max(bounds[0][0], min(bounds[0][1], x))
            y = max(bounds[1][0], min(bounds[1][1], y))
            self.position.append([x, y])
        
        self.position = np.array(self.position)
        
        # 初始化速度：随机小值
        self.velocity = np.random.uniform(-0.5, 0.5, self.position.shape)
        
        # 初始化个人最优
        self.pbest_position = np.copy(self.position)
        self.pbest_fitness = self.calculate_fitness()
        
    def calculate_fitness(self):
        """
        计算适应度（路径总长度，越小越好）
        
        返回：
            路径总长度
        """
        total_length = 0
        
        # 从起点到第一个中间点
        total_length += euclidean(self.start, self.position[0])
        
        # 中间点之间的距离
        for i in range(len(self.position) - 1):
            total_length += euclidean(self.position[i], self.position[i+1])
        
        # 从最后一个中间点到终点
        total_length += euclidean(self.position[-1], self.end)
        
        return total_length
    
    def update_velocity(self, gbest_position, w=0.5, c1=1.5, c2=1.5):
        """
        更新粒子速度
        
        参数：
            gbest_position: 全局最优位置
            w: 惯性权重（控制历史速度的影响）
            c1: 认知系数（个人经验的权重）
            c2: 社会系数（群体经验的权重）
        """
        # 随机因子
        r1 = np.random.random(self.position.shape)
        r2 = np.random.random(self.position.shape)
        
        # PSO速度更新公式：
        # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        cognitive_component = c1 * r1 * (self.pbest_position - self.position)
        social_component = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive_component + social_component
        
        # 限制速度范围，避免粒子飞太远
        max_velocity = 2.0
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
    
    def update_position(self, bounds):
        """
        更新粒子位置
        
        参数：
            bounds: 坐标边界
        """
        self.position += self.velocity
        
        # 限制位置在边界内
        for i in range(len(self.position)):
            self.position[i][0] = max(bounds[0][0], min(bounds[0][1], self.position[i][0]))
            self.position[i][1] = max(bounds[1][0], min(bounds[1][1], self.position[i][1]))
        
        # 计算新的适应度
        new_fitness = self.calculate_fitness()
        
        # 更新个人最优
        if new_fitness < self.pbest_fitness:
            self.pbest_position = np.copy(self.position)
            self.pbest_fitness = new_fitness
    
    def get_full_path(self):
        """
        获取完整路径（包括起点和终点）
        
        返回：
            full_path: 完整路径点列表
        """
        full_path = [self.start]
        for point in self.position:
            full_path.append(tuple(point))
        full_path.append(self.end)
        return full_path

class PSOOptimizer:
    """
    PSO优化器类
    
    核心算法流程：
    1. 初始化粒子群
    2. 计算每个粒子的适应度
    3. 更新全局最优
    4. 更新每个粒子的速度和位置
    5. 重复步骤2-4，直到达到最大迭代次数或收敛
    """
    
    def __init__(self, waypoints, bounds, num_particles=30, max_iter=100):
        """
        初始化PSO优化器
        
        参数：
            waypoints: 原始路径点
            bounds: 坐标边界
            num_particles: 粒子数量
            max_iter: 最大迭代次数
        """
        self.waypoints = waypoints
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        
        # 初始化粒子群
        self.particles = []
        for _ in range(num_particles):
            particle = Particle(waypoints, bounds)
            self.particles.append(particle)
        
        # 初始化全局最优
        self.gbest_position = None
        self.gbest_fitness = float('inf')
        self.update_gbest()
    
    def update_gbest(self):
        """
        更新全局最优位置
        """
        for particle in self.particles:
            if particle.pbest_fitness < self.gbest_fitness:
                self.gbest_fitness = particle.pbest_fitness
                self.gbest_position = np.copy(particle.pbest_position)
    
    def optimize(self):
        """
        执行PSO优化
        
        返回：
            best_path: 优化后的最优路径
            fitness_history: 每代最优适应度历史
        """
        fitness_history = []
        
        print(f"开始PSO优化，粒子数: {self.num_particles}，迭代次数: {self.max_iter}")
        print(f"初始路径长度: {self.particles[0].calculate_fitness():.2f}")
        
        for iteration in range(self.max_iter):
            # 更新每个粒子
            for particle in self.particles:
                particle.update_velocity(self.gbest_position)
                particle.update_position(self.bounds)
            
            # 更新全局最优
            self.update_gbest()
            
            # 记录历史最优
            fitness_history.append(self.gbest_fitness)
            
            # 打印进度
            if (iteration + 1) % 20 == 0:
                print(f"迭代 {iteration+1}/{self.max_iter}: 最优路径长度 = {self.gbest_fitness:.2f}")
        
        print(f"优化完成！最终路径长度: {self.gbest_fitness:.2f}")
        
        # 构建最优路径
        best_particle = None
        for particle in self.particles:
            if particle.pbest_fitness == self.gbest_fitness:
                best_particle = particle
                break
        
        return best_particle.get_full_path(), fitness_history

# ==============================================================================
# 第三步：可视化优化结果
# ==============================================================================

def plot_comparison(original_path, optimized_path, fitness_history, particle_history=None):
    """
    可视化原始路径和优化后的路径对比
    
    参数：
        original_path: 原始A*路径
        optimized_path: PSO优化后的路径
        fitness_history: 适应度历史
        particle_history: 粒子位置历史（用于动画）
    """
    # 提取坐标
    orig_x = [p[0] for p in original_path]
    orig_y = [p[1] for p in original_path]
    opt_x = [p[0] for p in optimized_path]
    opt_y = [p[1] for p in optimized_path]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 路径对比图
    ax1.plot(orig_x, orig_y, 'r--o', label='原始A*路径', linewidth=2, markersize=8)
    ax1.plot(opt_x, opt_y, 'b-o', label='PSO优化路径', linewidth=2, markersize=8)
    ax1.scatter(orig_x[0], orig_y[0], c='green', s=150, marker='o', zorder=5, label='起点')
    ax1.scatter(orig_x[-1], orig_y[-1], c='red', s=150, marker='*', zorder=5, label='终点')
    ax1.set_title('A*路径 vs PSO优化路径', fontsize=14)
    ax1.set_xlabel('X坐标', fontsize=12)
    ax1.set_ylabel('Y坐标', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True)
    
    # 适应度进化曲线
    ax2.plot(fitness_history, 'b-', linewidth=2)
    ax2.set_title('PSO适应度进化曲线', fontsize=14)
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('路径长度', fontsize=12)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('./PSO路径优化对比.png', dpi=150)
    plt.close()
    print("路径对比图已保存为 'PSO路径优化对比.png'")
    
    # 可视化2：长度对比柱状图
    orig_length = sum(euclidean(original_path[i], original_path[i+1]) for i in range(len(original_path)-1))
    opt_length = sum(euclidean(optimized_path[i], optimized_path[i+1]) for i in range(len(optimized_path)-1))
    
    plt.figure(figsize=(6, 5))
    plt.bar(['原始A*路径', 'PSO优化路径'], [orig_length, opt_length], color=['red', 'blue'])
    plt.title('路径长度对比', fontsize=14)
    plt.ylabel('长度', fontsize=12)
    plt.text(0, orig_length + 0.5, f'{orig_length:.2f}', ha='center', fontsize=12)
    plt.text(1, opt_length + 0.5, f'{opt_length:.2f}', ha='center', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('./PSO路径长度对比.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("长度对比图已保存为 'PSO路径长度对比.png'")
    
    # 可视化3：PSO粒子群动画（GIF）
    if particle_history:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(orig_x, orig_y, 'r--o', label='原始A*路径', linewidth=2, markersize=8, alpha=0.5)
        best_line, = ax.plot([], [], 'b-', linewidth=2.5, label='当前最优路径')
        particles_scatter = ax.scatter([], [], c='orange', s=50, label='粒子群')
        ax.scatter(orig_x[0], orig_y[0], c='green', s=150, marker='o', label='起点')
        ax.scatter(orig_x[-1], orig_y[-1], c='red', s=150, marker='*', label='终点')
        ax.set_title('PSO粒子群优化过程', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        def animate_pso(i):
            positions, gbest = particle_history[i]
            particles_scatter.set_offsets(positions.reshape(-1, 2))
            best_line.set_data(gbest[:, 0], gbest[:, 1])
            return particles_scatter, best_line
        
        anim = animation.FuncAnimation(fig, animate_pso, frames=min(50, len(particle_history)), 
                                       interval=150, blit=True)
        anim.save('./PSO粒子群动画.gif', writer='pillow', dpi=100)
        plt.close()
        print("粒子群动画已保存为 'PSO粒子群动画.gif'")

# ==============================================================================
# 第四步：主程序
# ==============================================================================

def main():
    """
    主函数：演示PSO路径优化流程
    """
    print("=" * 60)
    print("PSO路径优化演示 - 浙江大学高飞无人机团队")
    print("=" * 60)
    
    # --------------------------
    # 步骤1: 获取原始A*路径
    # --------------------------
    print("\n步骤1: 获取原始A*路径")
    original_path = generate_astar_path()
    print(f"  原始路径点: {original_path}")
    original_length = sum(euclidean(original_path[i], original_path[i+1]) for i in range(len(original_path)-1))
    print(f"  原始路径长度: {original_length:.2f}")
    
    # --------------------------
    # 步骤2: 设置优化参数
    # --------------------------
    print("\n步骤2: 设置PSO参数")
    bounds = [(0, 20), (0, 15)]  # X: 0-20, Y: 0-15
    num_particles = 30
    max_iter = 100
    print(f"  坐标边界: {bounds}")
    print(f"  粒子数量: {num_particles}")
    print(f"  最大迭代次数: {max_iter}")
    
    # --------------------------
    # 步骤3: 运行PSO优化
    # --------------------------
    print("\n步骤3: 运行PSO优化")
    optimizer = PSOOptimizer(original_path, bounds, num_particles, max_iter)
    optimized_path, fitness_history = optimizer.optimize()
    
    # --------------------------
    # 步骤4: 计算优化后的路径长度
    # --------------------------
    optimized_length = sum(euclidean(optimized_path[i], optimized_path[i+1]) for i in range(len(optimized_path)-1))
    improvement = ((original_length - optimized_length) / original_length) * 100
    print(f"\n步骤4: 优化结果分析")
    print(f"  原始路径长度: {original_length:.2f}")
    print(f"  优化后路径长度: {optimized_length:.2f}")
    print(f"  路径缩短: {improvement:.2f}%")
    print(f"  优化后的路径点: {optimized_path}")
    
    # --------------------------
    # 步骤5: 可视化
    # --------------------------
    print("\n步骤5: 可视化结果")
    plot_comparison(original_path, optimized_path, fitness_history)
    
    print("\n" + "=" * 60)
    print("PSO路径优化演示完成！")
    print("接下来学习：B-spline轨迹平滑")

if __name__ == '__main__':
    main()