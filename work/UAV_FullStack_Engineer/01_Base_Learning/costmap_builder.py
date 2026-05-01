#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浙大高飞无人机团队 | 代价地图构建模块
教学目标：掌握激光点云到代价地图的转换
对应论文：《Costmap 2D: A Generic Framework for Robot Navigation》
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.neighbors import KDTree

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CostmapBuilder:
    def __init__(self, map_size=50, resolution=0.5):
        """
        初始化代价地图构建器
        :param map_size: 地图尺寸（米）
        :param resolution: 栅格分辨率（米）
        """
        self.map_size = map_size
        self.resolution = resolution
        self.grid_size = int(map_size / resolution)
        self.costmap = np.zeros((self.grid_size, self.grid_size))
        
        # 定义代价等级
        self.FREE = 0
        self.OBSTACLE = 255
        self.INFLATED = 128
        self.UNKNOWN = 200
        
        # 膨胀半径（米）
        self.inflation_radius = 1.0
        self.inflation_cells = int(self.inflation_radius / resolution)
    
    def world_to_grid(self, x, y):
        """世界坐标转栅格坐标"""
        grid_x = int((x + self.map_size/2) / self.resolution)
        grid_y = int((y + self.map_size/2) / self.resolution)
        return (max(0, min(self.grid_size-1, grid_x)),
                max(0, min(self.grid_size-1, grid_y)))
    
    def grid_to_world(self, grid_x, grid_y):
        """栅格坐标转世界坐标"""
        x = grid_x * self.resolution - self.map_size/2
        y = grid_y * self.resolution - self.map_size/2
        return (x, y)
    
    def add_lidar_points(self, points):
        """
        添加激光点云数据
        :param points: 点云数据，形状为(N, 3)的numpy数组
        """
        for point in points:
            x, y, _ = point
            grid_x, grid_y = self.world_to_grid(x, y)
            self.costmap[grid_y, grid_x] = self.OBSTACLE
        
        # 执行膨胀操作
        self.inflate_obstacles()
    
    def inflate_obstacles(self):
        """膨胀障碍物（安全区域）"""
        inflated = self.costmap.copy()
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.costmap[y, x] == self.OBSTACLE:
                    # 对周围栅格进行膨胀
                    for dy in range(-self.inflation_cells, self.inflation_cells+1):
                        for dx in range(-self.inflation_cells, self.inflation_cells+1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                                distance = np.sqrt(dx**2 + dy**2) * self.resolution
                                if distance <= self.inflation_radius:
                                    if inflated[ny, nx] != self.OBSTACLE:
                                        # 距离越近，代价越高
                                        cost = int(200 - (distance/self.inflation_radius)*100)
                                        inflated[ny, nx] = max(inflated[ny, nx], cost)
        
        self.costmap = inflated
    
    def add_static_obstacle(self, x, y, radius=1.0):
        """添加静态障碍物"""
        grid_x, grid_y = self.world_to_grid(x, y)
        radius_cells = int(radius / self.resolution)
        
        for dy in range(-radius_cells, radius_cells+1):
            for dx in range(-radius_cells, radius_cells+1):
                ny, nx = grid_y + dy, grid_x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    distance = np.sqrt(dx**2 + dy**2) * self.resolution
                    if distance <= radius:
                        self.costmap[ny, nx] = self.OBSTACLE
        
        self.inflate_obstacles()
    
    def update_dynamic_obstacles(self, dynamic_points):
        """更新动态障碍物"""
        # 重置动态障碍物区域
        self.costmap = np.where(self.costmap >= 200, self.costmap, 0)
        
        # 添加新的动态障碍物
        self.add_lidar_points(dynamic_points)
    
    def visualize(self, path=None, save_path=None):
        """可视化代价地图"""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.costmap, cmap='gray', origin='lower', 
                   extent=[-self.map_size/2, self.map_size/2, 
                           -self.map_size/2, self.map_size/2])
        
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'r-', linewidth=2, label='规划路径')
            plt.scatter(path[0][0], path[0][1], c='green', s=100, label='起点')
            plt.scatter(path[-1][0], path[-1][1], c='blue', s=100, label='终点')
            plt.legend()
        
        plt.title('代价地图可视化')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.colorbar(label='代价值')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
    
    def save(self, file_path):
        """保存代价地图"""
        np.save(file_path, self.costmap)
        print(f"代价地图已保存至: {file_path}")
    
    def load(self, file_path):
        """加载代价地图"""
        self.costmap = np.load(file_path)
        print(f"代价地图已从 {file_path} 加载")

def generate_simulation_lidar():
    """生成模拟激光雷达数据"""
    np.random.seed(42)
    
    # 模拟障碍物点云
    points = []
    
    # 添加一些障碍物
    for _ in range(50):
        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        z = 0.0
        points.append([x, y, z])
    
    return np.array(points)

if __name__ == '__main__':
    print("代价地图构建演示")
    
    # 创建代价地图
    builder = CostmapBuilder(map_size=50, resolution=0.5)
    
    # 添加静态障碍物
    builder.add_static_obstacle(-10, 5, radius=2.0)
    builder.add_static_obstacle(10, -5, radius=3.0)
    builder.add_static_obstacle(0, 0, radius=1.5)
    
    # 添加模拟激光点云
    lidar_points = generate_simulation_lidar()
    builder.add_lidar_points(lidar_points)
    
    # 可视化
    builder.visualize(save_path='costmap_demo.png')
    
    # 保存
    builder.save('costmap.npy')
    
    print("演示完成！")