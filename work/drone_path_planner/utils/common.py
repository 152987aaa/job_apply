"""
浙大高飞无人机团队 | 通用工具函数
对应链接：数据读写、坐标转换、可视化、文件管理
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路点/轨迹读写 =====================
def save_waypoints(file_path, waypoints):
    """保存离散路点"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, waypoints, fmt="%.2f")
    print(f"路点已保存至: {file_path}")

def load_waypoints(file_path):
    """加载离散路点"""
    return np.loadtxt(file_path).tolist()

def save_trajectory(file_path, trajectory):
    """保存连续轨迹"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, trajectory, fmt="%.6f")
    print(f"轨迹已保存至: {file_path}")

def load_trajectory(file_path):
    """加载连续轨迹"""
    return np.loadtxt(file_path)

# ===================== 可视化工具 =====================
def plot_path(costmap, path, title="路径规划结果", save_path=None):
    """绘制地图+路径"""
    plt.figure(figsize=(8, 8))
    plt.imshow(costmap.T, cmap="gray_r", origin="lower")
    plt.plot([p[0] for p in path], [p[1] for p in path], "r-", linewidth=2, label="规划路径")
    plt.scatter(path[0][0], path[0][1], c="green", s=100, label="起点")
    plt.scatter(path[-1][0], path[-1][1], c="blue", s=100, label="终点")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"路径图已保存至: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_trajectory_comparison(waypoints, smoothed_traj, title="轨迹对比", save_path=None):
    """绘制原始路点与平滑轨迹对比"""
    plt.figure(figsize=(10, 8))
    plt.plot([p[0] for p in waypoints], [p[1] for p in waypoints], 
             "ro--", linewidth=1.5, markersize=8, label="原始路点")
    plt.plot(smoothed_traj[:, 0], smoothed_traj[:, 1], 
             "b-", linewidth=2, label="B-spline平滑轨迹")
    plt.scatter(waypoints[0][0], waypoints[0][1], c="green", s=150, marker="o", label="起点")
    plt.scatter(waypoints[-1][0], waypoints[-1][1], c="red", s=150, marker="*", label="终点")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"轨迹对比图已保存至: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_pso_convergence(fitness_history, title="PSO收敛曲线", save_path=None):
    """绘制PSO适应度进化曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, "b-", linewidth=2)
    plt.xlabel("迭代次数")
    plt.ylabel("路径长度 (m)")
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"收敛曲线图已保存至: {save_path}")
        plt.close()
    else:
        plt.show()

# ===================== 路径长度计算 =====================
def calculate_path_length(path):
    """计算路径总长度"""
    length = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        length += np.sqrt(dx**2 + dy**2)
    return length

# ===================== 坐标转换 =====================
def grid_to_world(grid_point, grid_resolution, origin=(0, 0)):
    """栅格坐标转世界坐标"""
    return (grid_point[0] * grid_resolution + origin[0],
            grid_point[1] * grid_resolution + origin[1])

def world_to_grid(world_point, grid_resolution, origin=(0, 0)):
    """世界坐标转栅格坐标"""
    return (int((world_point[0] - origin[0]) / grid_resolution),
            int((world_point[1] - origin[1]) / grid_resolution))