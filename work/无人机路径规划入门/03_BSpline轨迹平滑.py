#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-spline轨迹平滑 - 浙江大学高飞无人机团队教学材料

本文件实现了B-spline（B样条）曲线拟合算法，
用于将离散的路径点转换为连续光滑的轨迹。

教学目标：
1. 理解B-spline曲线的数学原理
2. 掌握控制点与基函数的概念
3. 学会使用B-spline进行轨迹平滑
4. 理解参数化和轨迹生成的过程
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.integrate import cumtrapz

# ==============================================================================
# 第一步：回顾PSO优化后的路径（模拟生成）
# ==============================================================================

def generate_optimized_path():
    """
    生成模拟的PSO优化后的路径点
    
    返回：
        waypoints: 路径点列表
    """
    waypoints = [
        (0, 0),
        (4.8, 2.3),
        (7.5, 4.8),
        (12.2, 6.1),
        (15.3, 9.8),
        (18.1, 12.2),
        (20, 15)
    ]
    return waypoints

# ==============================================================================
# 第二步：B-spline轨迹生成
# ==============================================================================

class BSplineTrajectory:
    """
    B-spline轨迹生成器
    
    B-spline曲线的优点：
    1. 局部性：修改一个控制点只影响局部曲线
    2. 光滑性：曲线具有连续的导数
    3. 灵活性：可以通过调整控制点灵活控制曲线形状
    """
    
    def __init__(self, waypoints, degree=3):
        """
        初始化B-spline轨迹生成器
        
        参数：
            waypoints: 路径点列表
            degree: B-spline曲线的阶数（通常使用3阶，即立方B-spline）
        """
        self.waypoints = np.array(waypoints)
        self.degree = degree
        
        # 提取坐标
        self.x = self.waypoints[:, 0]
        self.y = self.waypoints[:, 1]
        
        # 计算参数化参数t
        # 使用累积距离作为参数，确保曲线均匀分布
        self.t = self._compute_parameterization()
        
        # 创建B-spline插值器
        self.spline_x = None
        self.spline_y = None
        self._create_spline()
    
    def _compute_parameterization(self):
        """
        计算参数化参数t
        
        使用累积欧几里得距离作为参数，这样可以确保：
        1. 参数t与实际路径长度成正比
        2. 曲线在路径点之间均匀分布
        
        返回：
            t: 参数化参数数组
        """
        # 计算相邻点之间的距离
        distances = np.zeros(len(self.waypoints))
        for i in range(1, len(self.waypoints)):
            distances[i] = np.sqrt((self.x[i] - self.x[i-1])**2 + 
                                   (self.y[i] - self.y[i-1])**2)
        
        # 计算累积距离
        t = np.cumsum(distances)
        t = t / t[-1]  # 归一化到[0, 1]
        
        return t
    
    def _create_spline(self):
        """
        创建B-spline插值器
        
        使用scipy的make_interp_spline函数，这是一个方便的工具函数，
        可以自动计算B-spline的控制点和节点向量。
        """
        # 创建x坐标的B-spline
        self.spline_x = make_interp_spline(self.t, self.x, k=self.degree)
        
        # 创建y坐标的B-spline
        self.spline_y = make_interp_spline(self.t, self.y, k=self.degree)
    
    def get_trajectory(self, num_points=100):
        """
        生成平滑的轨迹点
        
        参数：
            num_points: 生成的轨迹点数量
        
        返回：
            trajectory: 平滑轨迹点数组，形状为(num_points, 2)
        """
        # 在参数t的范围内均匀采样
        t_eval = np.linspace(0, 1, num_points)
        
        # 计算轨迹点
        x_eval = self.spline_x(t_eval)
        y_eval = self.spline_y(t_eval)
        
        trajectory = np.column_stack((x_eval, y_eval))
        
        return trajectory
    
    def get_velocity(self, num_points=100):
        """
        计算轨迹的速度
        
        参数：
            num_points: 采样点数量
        
        返回：
            velocities: 速度数组，形状为(num_points, 2)
        """
        t_eval = np.linspace(0, 1, num_points)
        
        # 计算导数（速度）
        dx_dt = self.spline_x.derivative()(t_eval)
        dy_dt = self.spline_y.derivative()(t_eval)
        
        velocities = np.column_stack((dx_dt, dy_dt))
        
        return velocities
    
    def get_acceleration(self, num_points=100):
        """
        计算轨迹的加速度
        
        参数：
            num_points: 采样点数量
        
        返回：
            accelerations: 加速度数组，形状为(num_points, 2)
        """
        t_eval = np.linspace(0, 1, num_points)
        
        # 计算二阶导数（加速度）
        ddx_dt2 = self.spline_x.derivative(2)(t_eval)
        ddy_dt2 = self.spline_y.derivative(2)(t_eval)
        
        accelerations = np.column_stack((ddx_dt2, ddy_dt2))
        
        return accelerations
    
    def compute_path_length(self, num_points=1000):
        """
        计算平滑轨迹的总长度
        
        参数：
            num_points: 用于数值积分的采样点数量
        
        返回：
            length: 路径总长度
        """
        t_eval = np.linspace(0, 1, num_points)
        
        # 计算速度大小
        dx_dt = self.spline_x.derivative()(t_eval)
        dy_dt = self.spline_y.derivative()(t_eval)
        speed = np.sqrt(dx_dt**2 + dy_dt**2)
        
        # 使用数值积分计算路径长度
        length = cumtrapz(speed, t_eval, initial=0)[-1]
        
        return length

# ==============================================================================
# 第三步：轨迹可视化
# ==============================================================================

def plot_trajectory_comparison(waypoints, smooth_trajectory):
    """
    可视化原始路径点和平滑轨迹的对比
    
    参数：
        waypoints: 原始路径点
        smooth_trajectory: 平滑后的轨迹
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 轨迹对比图
    waypoints_np = np.array(waypoints)
    ax1.plot(waypoints_np[:, 0], waypoints_np[:, 1], 'r--o', 
             label='原始路径点', linewidth=2, markersize=10)
    ax1.plot(smooth_trajectory[:, 0], smooth_trajectory[:, 1], 'b-', 
             label='B-spline平滑轨迹', linewidth=2)
    ax1.scatter(waypoints[0][0], waypoints[0][1], c='green', s=150, marker='o', zorder=5, label='起点')
    ax1.scatter(waypoints[-1][0], waypoints[-1][1], c='red', s=150, marker='*', zorder=5, label='终点')
    
    # 标记路径点
    for i, (x, y) in enumerate(waypoints):
        ax1.annotate(f'P{i}', (x+0.3, y+0.3), fontsize=10)
    
    ax1.set_title('原始路径点 vs B-spline平滑轨迹')
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.legend()
    ax1.grid(True)
    
    # 轨迹细节放大图（中间部分）
    ax2.plot(smooth_trajectory[:, 0], smooth_trajectory[:, 1], 'b-', label='平滑轨迹')
    ax2.scatter(waypoints_np[:, 0], waypoints_np[:, 1], c='red', s=50, label='路径点')
    ax2.set_xlim(5, 15)
    ax2.set_ylim(4, 12)
    ax2.set_title('轨迹细节放大图')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('./BSpline轨迹平滑.png')
    plt.close()
    print("轨迹平滑图已保存为 'BSpline轨迹平滑.png'")

def plot_velocity_profile(velocities):
    """
    可视化速度分布
    
    参数：
        velocities: 速度数组
    """
    speed = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(speed, 'b-', linewidth=2)
    plt.title('B-spline轨迹速度分布')
    plt.xlabel('轨迹点索引')
    plt.ylabel('速度大小')
    plt.grid(True)
    plt.savefig('./BSpline速度分布.png')
    plt.close()
    print("速度分布图已保存为 'BSpline速度分布.png'")

# ==============================================================================
# 第四步：主程序
# ==============================================================================

def main():
    """
    主函数：演示B-spline轨迹平滑流程
    """
    print("=" * 60)
    print("B-spline轨迹平滑演示 - 浙江大学高飞无人机团队")
    print("=" * 60)
    
    # --------------------------
    # 步骤1: 获取PSO优化后的路径
    # --------------------------
    print("\n步骤1: 获取PSO优化后的路径")
    waypoints = generate_optimized_path()
    print(f"  路径点数量: {len(waypoints)}")
    print(f"  路径点坐标:")
    for i, (x, y) in enumerate(waypoints):
        print(f"    P{i}: ({x:.2f}, {y:.2f})")
    
    # --------------------------
    # 步骤2: 创建B-spline轨迹生成器
    # --------------------------
    print("\n步骤2: 创建B-spline轨迹生成器")
    degree = 3  # 立方B-spline
    trajectory_generator = BSplineTrajectory(waypoints, degree=degree)
    print(f"  B-spline阶数: {degree}阶（立方B-spline）")
    
    # --------------------------
    # 步骤3: 生成平滑轨迹
    # --------------------------
    print("\n步骤3: 生成平滑轨迹")
    num_points = 100
    smooth_trajectory = trajectory_generator.get_trajectory(num_points)
    print(f"  生成轨迹点数量: {num_points}")
    
    # --------------------------
    # 步骤4: 计算轨迹属性
    # --------------------------
    print("\n步骤4: 计算轨迹属性")
    
    # 计算路径长度
    original_length = sum(np.sqrt((waypoints[i][0]-waypoints[i+1][0])**2 + 
                                  (waypoints[i][1]-waypoints[i+1][1])**2) 
                         for i in range(len(waypoints)-1))
    smooth_length = trajectory_generator.compute_path_length()
    print(f"  原始路径长度: {original_length:.2f}")
    print(f"  平滑后路径长度: {smooth_length:.2f}")
    
    # 计算速度信息
    velocities = trajectory_generator.get_velocity(num_points)
    max_speed = np.max(np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2))
    avg_speed = np.mean(np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2))
    print(f"  最大速度: {max_speed:.2f}")
    print(f"  平均速度: {avg_speed:.2f}")
    
    # --------------------------
    # 步骤5: 可视化
    # --------------------------
    print("\n步骤5: 可视化结果")
    plot_trajectory_comparison(waypoints, smooth_trajectory)
    plot_velocity_profile(velocities)
    
    print("\n" + "=" * 60)
    print("B-spline轨迹平滑演示完成！")
    print("接下来学习：Gazebo无人机仿真与轨迹跟踪")

if __name__ == '__main__':
    main()