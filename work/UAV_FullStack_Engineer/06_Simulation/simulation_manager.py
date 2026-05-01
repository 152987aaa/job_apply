#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浙大高飞无人机团队 | 仿真验证模块
教学目标：掌握Gazebo、PX4 SITL仿真
对应论文：《Gazebo Robot Simulation Handbook》
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../02_Path_Planning')
sys.path.append('../01_Base_Learning')
from path_planners import AStarPlanner, DWAPlanner
from uav_dynamics import QuadrotorDynamics

class GazeboInterface:
    """Gazebo仿真接口"""
    
    def __init__(self, drone_model='firefly'):
        """
        初始化Gazebo接口
        :param drone_model: 无人机模型名称
        """
        self.drone_model = drone_model
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x, y, z, roll, pitch, yaw
        self.current_velocity = np.array([0.0, 0.0, 0.0])
        self.simulation_time = 0.0
        
        # 模拟Gazebo环境
        self.obstacles = []
        self.ground_truth_path = []
    
    def spawn_drone(self, x=0, y=0, z=1):
        """在Gazebo中生成无人机"""
        print(f"生成无人机 {self.drone_model} 于 ({x}, {y}, {z})")
        self.current_pose[:3] = [x, y, z]
    
    def set_velocity(self, vx, vy, vz):
        """设置无人机速度"""
        self.current_velocity = np.array([vx, vy, vz])
    
    def step(self, dt=0.01):
        """执行仿真步"""
        # 更新位置
        self.current_pose[:3] += self.current_velocity * dt
        self.simulation_time += dt
        
        # 记录轨迹
        self.ground_truth_path.append(self.current_pose[:3].copy())
    
    def get_ground_truth(self):
        """获取真实位姿"""
        return {
            'position': self.current_pose[:3],
            'orientation': self.current_pose[3:],
            'velocity': self.current_velocity,
            'time': self.simulation_time
        }
    
    def add_obstacle(self, x, y, z, radius=1.0):
        """添加障碍物"""
        self.obstacles.append({'x': x, 'y': y, 'z': z, 'radius': radius})
        print(f"添加障碍物于 ({x}, {y}, {z})")

class PX4SITLInterface:
    """PX4软件在环仿真接口"""
    
    def __init__(self):
        """初始化PX4 SITL接口"""
        self.armed = False
        self.mode = 'STABILIZE'
        self.attitude = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.throttle = 0.0
        
        # MAVLink消息计数器
        self.message_count = 0
    
    def arm(self):
        """解锁无人机"""
        self.armed = True
        print("PX4: 无人机已解锁")
    
    def disarm(self):
        """上锁无人机"""
        self.armed = False
        print("PX4: 无人机已上锁")
    
    def set_mode(self, mode):
        """设置飞行模式"""
        valid_modes = ['STABILIZE', 'ALT_HOLD', 'AUTO', 'GUIDED']
        if mode in valid_modes:
            self.mode = mode
            print(f"PX4: 切换到 {mode} 模式")
        else:
            print(f"PX4: 无效模式 {mode}")
    
    def send_attitude_command(self, roll, pitch, yaw_rate, throttle):
        """发送姿态指令"""
        if not self.armed:
            print("PX4: 无人机未解锁")
            return
        
        self.attitude = np.array([roll, pitch, self.attitude[2] + yaw_rate * 0.01])
        self.throttle = max(0.0, min(1.0, throttle))
        
        self.message_count += 1
        if self.message_count % 100 == 0:
            print(f"PX4: 已发送 {self.message_count} 条指令")
    
    def send_position_command(self, x, y, z, yaw=0.0):
        """发送位置指令"""
        if not self.armed:
            print("PX4: 无人机未解锁")
            return
        
        print(f"PX4: 目标位置 ({x}, {y}, {z}), 偏航 {yaw}")

class SimulationManager:
    """仿真管理器"""
    
    def __init__(self):
        """初始化仿真管理器"""
        self.gazebo = GazeboInterface()
        self.px4 = PX4SITLInterface()
        self.path_planner = None
        self.trajectory_tracker = None
        
        # 数据记录
        self.log_data = {
            'time': [],
            'position': [],
            'velocity': [],
            'command': []
        }
    
    def setup_simulation(self, scenario='default'):
        """设置仿真场景"""
        print(f"设置仿真场景: {scenario}")
        
        if scenario == 'default':
            # 默认场景：简单悬停
            self.gazebo.spawn_drone(0, 0, 2)
            self.px4.set_mode('ALT_HOLD')
        
        elif scenario == 'navigation':
            # 导航场景
            self.gazebo.spawn_drone(0, 0, 1)
            
            # 添加障碍物
            self.gazebo.add_obstacle(5, 5, 0, radius=2.0)
            self.gazebo.add_obstacle(10, 10, 0, radius=1.5)
            
            # 初始化路径规划器
            grid = np.zeros((20, 20))
            grid[4:7, 4:7] = 1
            grid[9:12, 9:12] = 1
            self.path_planner = AStarPlanner(grid)
    
    def run_simulation(self, duration=10.0, dt=0.01):
        """运行仿真"""
        print(f"开始仿真，时长 {duration}s")
        
        # 解锁无人机
        self.px4.arm()
        
        num_steps = int(duration / dt)
        
        for step in range(num_steps):
            # 更新Gazebo
            self.gazebo.step(dt)
            
            # 记录数据
            ground_truth = self.gazebo.get_ground_truth()
            self.log_data['time'].append(ground_truth['time'])
            self.log_data['position'].append(ground_truth['position'].copy())
            self.log_data['velocity'].append(ground_truth['velocity'].copy())
            
            # 简单控制：悬停
            self.px4.send_attitude_command(0, 0, 0, 0.5)
            self.gazebo.set_velocity(0, 0, 0)
            
            # 进度显示
            if step % 100 == 0:
                progress = (step / num_steps) * 100
                print(f"仿真进度: {progress:.1f}%")
        
        # 上锁无人机
        self.px4.disarm()
        
        print("仿真完成")
        self.save_logs()
    
    def save_logs(self):
        """保存仿真日志"""
        np.save('simulation_logs/time.npy', np.array(self.log_data['time']))
        np.save('simulation_logs/position.npy', np.array(self.log_data['position']))
        np.save('simulation_logs/velocity.npy', np.array(self.log_data['velocity']))
        print("仿真日志已保存")
    
    def visualize_results(self):
        """可视化仿真结果"""
        positions = np.array(self.log_data['position'])
        
        plt.figure(figsize=(12, 6))
        
        # 位置轨迹
        plt.subplot(1, 2, 1)
        plt.plot(self.log_data['time'], positions[:, 0], label='X')
        plt.plot(self.log_data['time'], positions[:, 1], label='Y')
        plt.plot(self.log_data['time'], positions[:, 2], label='Z')
        plt.xlabel('时间 (s)')
        plt.ylabel('位置 (m)')
        plt.title('无人机位置轨迹')
        plt.legend()
        plt.grid(True)
        
        # 2D轨迹
        plt.subplot(1, 2, 2)
        plt.plot(positions[:, 0], positions[:, 1], 'b-')
        plt.scatter(positions[0, 0], positions[0, 1], c='green', label='起点')
        plt.scatter(positions[-1, 0], positions[-1, 1], c='red', label='终点')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('2D轨迹')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('simulation_results.png')
        plt.close()
        print("仿真结果图已保存")

def run_simulation():
    """运行仿真示例"""
    sim_manager = SimulationManager()
    
    # 设置场景
    sim_manager.setup_simulation('navigation')
    
    # 运行仿真
    sim_manager.run_simulation(duration=10.0)
    
    # 可视化结果
    sim_manager.visualize_results()

if __name__ == '__main__':
    run_simulation()