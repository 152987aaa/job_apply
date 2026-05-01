#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浙大高飞无人机团队 | 全局-局部协同规划模块
教学目标：掌握全局规划与局部规划的协同逻辑
对应论文：《Global-Local Path Planning for Mobile Robots》
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../02_Path_Planning')
from path_planners import AStarPlanner, DWAPlanner

class CooperativePlanner:
    """全局-局部协同规划器"""
    
    def __init__(self, global_grid, local_bounds=[(-5, 5), (-5, 5)]):
        """
        初始化协同规划器
        :param global_grid: 全局代价地图
        :param local_bounds: 局部规划范围
        """
        # 全局规划器
        self.global_planner = AStarPlanner(global_grid)
        
        # 局部规划器
        self.local_planner = DWAPlanner(max_speed=2.0, max_accel=1.0)
        
        # 全局路径
        self.global_path = None
        self.path_index = 0
        
        # 当前状态
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        
        # 动态障碍物
        self.dynamic_obstacles = []
    
    def set_global_path(self, start, goal):
        """设置全局路径"""
        self.global_path = self.global_planner.plan(start, goal)
        self.path_index = 0
        
        if self.global_path:
            print(f"全局路径规划完成，共 {len(self.global_path)} 个路点")
        else:
            print("全局路径规划失败")
    
    def get_local_goal(self, look_ahead=5):
        """获取局部目标点"""
        if not self.global_path:
            return None
        
        target_index = min(self.path_index + look_ahead, len(self.global_path) - 1)
        return self.global_path[target_index]
    
    def update_dynamic_obstacles(self, obstacles):
        """更新动态障碍物信息"""
        self.dynamic_obstacles = obstacles
    
    def update_pose(self, pose):
        """更新当前位姿"""
        self.current_pose = pose
        
        # 更新路径索引
        if self.global_path:
            while self.path_index < len(self.global_path) - 1:
                dist = np.hypot(
                    self.current_pose[0] - self.global_path[self.path_index][0],
                    self.current_pose[1] - self.global_path[self.path_index][1]
                )
                if dist < 0.5:
                    self.path_index += 1
                else:
                    break
    
    def plan(self):
        """执行协同规划"""
        if not self.global_path:
            return (0.0, 0.0)
        
        # 获取局部目标
        local_goal = self.get_local_goal()
        if local_goal is None:
            return (0.0, 0.0)
        
        # 使用DWA进行局部避障
        v, w = self.local_planner.plan(
            self.current_pose,
            local_goal,
            self.dynamic_obstacles
        )
        
        return (v, w)
    
    def is_goal_reached(self, goal, threshold=1.0):
        """检查是否到达终点"""
        dist = np.hypot(self.current_pose[0] - goal[0], self.current_pose[1] - goal[1])
        return dist < threshold

class MPCController:
    """模型预测控制器"""
    
    def __init__(self, horizon=10, dt=0.1):
        """
        初始化MPC控制器
        :param horizon: 预测步数
        :param dt: 时间步长
        """
        self.horizon = horizon
        self.dt = dt
        self.Q = np.diag([1.0, 1.0, 0.1])  # 状态权重
        self.R = np.diag([0.1, 0.1])        # 控制权重
    
    def predict_state(self, x, u):
        """
        预测下一状态
        :param x: 当前状态 [x, y, theta]
        :param u: 控制输入 [v, w]
        :return: 下一状态
        """
        x_next = np.array([
            x[0] + u[0] * np.cos(x[2]) * self.dt,
            x[1] + u[0] * np.sin(x[2]) * self.dt,
            x[2] + u[1] * self.dt
        ])
        return x_next
    
    def compute_cost(self, x, u, ref_traj):
        """
        计算代价
        :param x: 当前状态
        :param u: 控制序列
        :param ref_traj: 参考轨迹
        :return: 总代价
        """
        cost = 0.0
        current_x = x
        
        for i in range(self.horizon):
            # 状态代价
            if i < len(ref_traj):
                error = current_x - ref_traj[i]
                cost += error @ self.Q @ error
            
            # 控制代价
            cost += u[i] @ self.R @ u[i]
            
            # 预测下一状态
            current_x = self.predict_state(current_x, u[i])
        
        return cost
    
    def optimize(self, x, ref_traj):
        """
        优化控制序列
        :param x: 当前状态
        :param ref_traj: 参考轨迹
        :return: 最优控制输入
        """
        # 简化实现：使用随机搜索
        best_cost = float('inf')
        best_u = np.zeros(2)
        
        for v in np.linspace(-2.0, 2.0, 11):
            for w in np.linspace(-1.0, 1.0, 11):
                u_seq = np.tile([v, w], (self.horizon, 1))
                cost = self.compute_cost(x, u_seq, ref_traj)
                if cost < best_cost:
                    best_cost = cost
                    best_u = np.array([v, w])
        
        return best_u

def simulate_cooperative_planning():
    """模拟协同规划"""
    # 创建全局地图
    grid = np.zeros((30, 30))
    grid[10:15, 10:15] = 1
    grid[15:20, 5:10] = 1
    
    # 初始化协同规划器
    planner = CooperativePlanner(grid)
    
    # 设置全局路径
    start = (2, 2)
    goal = (28, 28)
    planner.set_global_path(start, goal)
    
    if not planner.global_path:
        print("无法规划全局路径")
        return
    
    # 模拟执行
    planner.current_pose = np.array([2.0, 2.0, 0.0])
    path = [(2.0, 2.0)]
    
    for _ in range(200):
        # 更新动态障碍物（模拟移动障碍物）
        dynamic_obs = [
            (15 + np.cos(_*0.1)*2, 12 + np.sin(_*0.1)*2)
        ]
        planner.update_dynamic_obstacles(dynamic_obs)
        
        # 执行规划
        v, w = planner.plan()
        
        # 更新位姿
        dt = 0.1
        planner.current_pose[0] += v * np.cos(planner.current_pose[2]) * dt
        planner.current_pose[1] += v * np.sin(planner.current_pose[2]) * dt
        planner.current_pose[2] += w * dt
        
        path.append((planner.current_pose[0], planner.current_pose[1]))
        
        # 检查是否到达终点
        if planner.is_goal_reached(goal):
            print("到达终点！")
            break
    
    # 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.T, cmap='gray', origin='lower')
    plt.plot([p[0] for p in planner.global_path], [p[1] for p in planner.global_path], 
             'r--', label='全局路径')
    plt.plot([p[0] for p in path], [p[1] for p in path], 'b-', label='实际轨迹')
    plt.scatter(start[0], start[1], c='green', s=100, label='起点')
    plt.scatter(goal[0], goal[1], c='red', s=100, label='终点')
    plt.legend()
    plt.title('全局-局部协同规划仿真')
    plt.savefig('cooperative_planning.png')
    plt.close()
    
    print("协同规划仿真完成！")

if __name__ == '__main__':
    simulate_cooperative_planning()