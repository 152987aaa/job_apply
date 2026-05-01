"""
浙大高飞无人机团队 | B-spline轨迹平滑器
【严格对应链接步骤3】
功能：PSO优化路点 → B-spline平滑 → 生成连续轨迹 → 输出轨迹文件
教学重点：B-spline原理、参数化方法、轨迹导数计算
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import make_interp_spline, BSpline
from scipy.integrate import cumulative_trapezoid
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import load_waypoints, save_trajectory, plot_trajectory_comparison

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 参数设置（可调参学习）=====================
SPLINE_DEGREE = 3        # B-spline阶数（3=立方B-spline）
NUM_SAMPLES = 100        # 轨迹采样点数

# ===================== B-spline轨迹生成器 =====================
class BSplineGenerator:
    def __init__(self, waypoints, degree=3):
        """
        初始化B-spline轨迹生成器
        :param waypoints: 离散路点列表
        :param degree: B-spline阶数
        """
        self.waypoints = np.array(waypoints)
        self.degree = degree
        
        # 提取坐标
        self.x = self.waypoints[:, 0]
        self.y = self.waypoints[:, 1]
        
        # 计算参数化参数t（累积距离参数化）
        self.t = self._compute_parameterization()
        
        # 创建B-spline插值器
        self.spline_x = None
        self.spline_y = None
        self._create_spline()
    
    def _compute_parameterization(self):
        """
        计算参数化参数t
        使用累积欧几里得距离作为参数，确保曲线均匀分布
        :return: 参数化参数数组
        """
        # 计算相邻点之间的距离
        distances = np.zeros(len(self.waypoints))
        for i in range(1, len(self.waypoints)):
            distances[i] = np.hypot(self.x[i] - self.x[i-1],
                                   self.y[i] - self.y[i-1])
        
        # 计算累积距离并归一化到[0, 1]
        t = np.cumsum(distances)
        t = t / t[-1]
        
        return t
    
    def _create_spline(self):
        """创建B-spline插值器"""
        # 创建x坐标的B-spline
        self.spline_x = make_interp_spline(self.t, self.x, k=self.degree)
        
        # 创建y坐标的B-spline
        self.spline_y = make_interp_spline(self.t, self.y, k=self.degree)
    
    def generate_trajectory(self, num_points=100):
        """
        生成平滑轨迹
        :param num_points: 采样点数
        :return: 轨迹数组 (num_points, 2)
        """
        # 在参数范围内均匀采样
        t_eval = np.linspace(0, 1, num_points)
        
        # 计算轨迹点
        x_eval = self.spline_x(t_eval)
        y_eval = self.spline_y(t_eval)
        
        trajectory = np.column_stack((x_eval, y_eval))
        
        return trajectory
    
    def compute_velocity(self, num_points=100):
        """
        计算轨迹速度
        :param num_points: 采样点数
        :return: 速度数组 (num_points, 2)
        """
        t_eval = np.linspace(0, 1, num_points)
        
        # 计算导数（速度）
        dx_dt = self.spline_x.derivative()(t_eval)
        dy_dt = self.spline_y.derivative()(t_eval)
        
        velocities = np.column_stack((dx_dt, dy_dt))
        
        return velocities
    
    def compute_acceleration(self, num_points=100):
        """
        计算轨迹加速度
        :param num_points: 采样点数
        :return: 加速度数组 (num_points, 2)
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
        :param num_points: 采样点数（用于数值积分）
        :return: 路径长度
        """
        t_eval = np.linspace(0, 1, num_points)
        
        # 计算速度大小
        dx_dt = self.spline_x.derivative()(t_eval)
        dy_dt = self.spline_y.derivative()(t_eval)
        speed = np.sqrt(dx_dt**2 + dy_dt**2)
        
        # 使用数值积分计算路径长度
        length = cumulative_trapezoid(speed, t_eval, initial=0)[-1]
        
        return length

# ===================== 主函数 =====================
def main():
    print("=" * 60)
    print("B-spline轨迹平滑器")
    print("浙大高飞无人机团队")
    print("=" * 60)
    
    # 步骤1：加载PSO优化后的路点
    print("\n【步骤1】加载PSO优化路点")
    try:
        waypoints = load_waypoints("../data/output/pso_optimized_waypoints.txt")
        print(f"Loaded {len(waypoints)} waypoints successfully")
    except Exception as e:
        print(f"Failed to load PSO waypoints: {e}")
        print("Please run PSO optimizer first")
        return
    
    # 步骤2：创建B-spline生成器
    print("\n【步骤2】创建B-spline轨迹生成器")
    generator = BSplineGenerator(waypoints, degree=SPLINE_DEGREE)
    print(f"B-spline阶数: {SPLINE_DEGREE}")
    
    # 步骤3：生成平滑轨迹
    print("\n【步骤3】生成平滑轨迹")
    trajectory = generator.generate_trajectory(NUM_SAMPLES)
    print(f"生成 {NUM_SAMPLES} 个轨迹点")
    
    # 步骤4：计算轨迹属性
    print("\n【步骤4】计算轨迹属性")
    
    # 路径长度
    original_length = 0
    for i in range(len(waypoints) - 1):
        original_length += np.hypot(waypoints[i+1][0] - waypoints[i][0],
                                   waypoints[i+1][1] - waypoints[i][1])
    smooth_length = generator.compute_path_length()
    print(f"原始路点路径长度: {original_length:.2f} 米")
    print(f"平滑后路径长度: {smooth_length:.2f} 米")
    
    # 速度信息
    velocities = generator.compute_velocity(NUM_SAMPLES)
    speed = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    print(f"最大速度: {np.max(speed):.2f}")
    print(f"平均速度: {np.mean(speed):.2f}")
    
    # 加速度信息
    accelerations = generator.compute_acceleration(NUM_SAMPLES)
    acc_magnitude = np.sqrt(accelerations[:, 0]**2 + accelerations[:, 1]**2)
    print(f"最大加速度: {np.max(acc_magnitude):.2f}")
    
    # 步骤5：保存轨迹
    print("\n【步骤5】保存轨迹数据")
    save_trajectory("../data/output/bspline_trajectory.txt", trajectory)
    
    # 步骤6：可视化
    print("\n【步骤6】可视化轨迹")
    plot_trajectory_comparison(waypoints, trajectory, 
                               "B-spline轨迹平滑结果", 
                               "../data/output/bspline_trajectory.png")
    
    # 可视化2：速度分布图
    plt.figure(figsize=(10, 5))
    plt.plot(speed, 'b-', linewidth=2)
    plt.title('B-spline轨迹速度分布', fontsize=14)
    plt.xlabel('采样点索引', fontsize=12)
    plt.ylabel('速度', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('../data/output/bspline_velocity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("速度分布图已保存")
    
    # 可视化3：加速度分布图
    plt.figure(figsize=(10, 5))
    plt.plot(acc_magnitude, 'r-', linewidth=2)
    plt.title('B-spline轨迹加速度分布', fontsize=14)
    plt.xlabel('采样点索引', fontsize=12)
    plt.ylabel('加速度', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('../data/output/bspline_acceleration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("加速度分布图已保存")
    
    # 可视化4：轨迹3D视图（带速度颜色）
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    sc = ax.scatter(trajectory[:, 0], trajectory[:, 1], c=speed, cmap='viridis', s=50)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=2)
    ax.scatter(waypoints[0][0], waypoints[0][1], c='green', s=150, marker='o', label='起点')
    ax.scatter(waypoints[-1][0], waypoints[-1][1], c='blue', s=150, marker='*', label='终点')
    plt.colorbar(sc, label='速度')
    plt.title('B-spline轨迹速度分布（颜色编码）', fontsize=14)
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('../data/output/bspline_velocity_color.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("速度颜色编码图已保存")
    
    # 可视化5：轨迹动画（GIF）
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([p[0] for p in waypoints], [p[1] for p in waypoints], 'b--', linewidth=2, label='原始路点')
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=2, alpha=0.5)
    drone, = ax.plot([], [], 'ro', markersize=15, label='无人机')
    path_line, = ax.plot([], [], 'r-', linewidth=3)
    ax.scatter(waypoints[0][0], waypoints[0][1], c='green', s=150, marker='o', label='起点')
    ax.scatter(waypoints[-1][0], waypoints[-1][1], c='blue', s=150, marker='*', label='终点')
    ax.set_title('B-spline轨迹跟踪动画', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 26)
    ax.set_ylim(-1, 26)
    
    def animate_bspline(i):
        path_line.set_data(trajectory[:i+1, 0], trajectory[:i+1, 1])
        drone.set_data([trajectory[i, 0]], [trajectory[i, 1]])
        return path_line, drone
    
    anim = animation.FuncAnimation(fig, animate_bspline, frames=NUM_SAMPLES, interval=50, blit=True)
    anim.save('../data/output/bspline_animation.gif', writer='pillow', dpi=100)
    plt.close()
    print("轨迹跟踪动画GIF已保存")
    
    # 保存速度和加速度数据
    np.savetxt("../data/output/trajectory_velocity.txt", velocities, fmt="%.6f")
    np.savetxt("../data/output/trajectory_acceleration.txt", accelerations, fmt="%.6f")
    
    print("\n[B-spline Done] Trajectory smoothing completed!")
    print("Output files:")
    print("  - data/output/bspline_trajectory.txt: 平滑轨迹（供Gazebo使用）")
    print("  - data/output/trajectory_velocity.txt: 速度数据")
    print("  - data/output/trajectory_acceleration.txt: 加速度数据")
    print("  - data/output/bspline_trajectory.png: 轨迹对比图")
    print("  - data/output/bspline_velocity.png: 速度分布图")
    print("  - data/output/bspline_acceleration.png: 加速度分布图")
    print("  - data/output/bspline_velocity_color.png: 速度颜色编码图")
    print("  - data/output/bspline_animation.gif: 轨迹跟踪动画")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()