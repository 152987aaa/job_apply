#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浙大高飞无人机团队 | 无人机动力学仿真模块
教学目标：掌握无人机运动学和动力学建模
对应论文：《UAV Dynamics: Modeling and Control》
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class QuadrotorDynamics:
    def __init__(self, mass=1.0, arm_length=0.25, Ixx=0.01, Iyy=0.01, Izz=0.02):
        """
        初始化四旋翼无人机动力学模型
        :param mass: 质量（kg）
        :param arm_length: 臂长（m）
        :param Ixx, Iyy, Izz: 转动惯量（kg·m²）
        """
        self.mass = mass
        self.L = arm_length
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        
        # 重力加速度
        self.g = 9.81
        
        # 状态向量: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.state = np.zeros(12)
        
        # 电机配置矩阵
        self.B = np.array([
            [1, 1, 1, 1],
            [0, self.L, 0, -self.L],
            [-self.L, 0, self.L, 0],
            [1, -1, 1, -1]
        ])
        
        # 力矩系数
        self.k_t = 1.0  # 推力系数
        self.k_d = 0.01  # 阻力系数
    
    def set_state(self, x, y, z, vx=0, vy=0, vz=0, 
                  roll=0, pitch=0, yaw=0, p=0, q=0, r=0):
        """设置无人机状态"""
        self.state = np.array([x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r])
    
    def get_state(self):
        """获取无人机状态"""
        return {
            'position': self.state[:3],
            'velocity': self.state[3:6],
            'attitude': self.state[6:9],
            'angular_velocity': self.state[9:]
        }
    
    def rotation_matrix(self):
        """计算旋转矩阵（从机体坐标系到世界坐标系）"""
        roll, pitch, yaw = self.state[6], self.state[7], self.state[8]
        
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        return R
    
    def compute_forces(self, motor_thrusts):
        """
        计算无人机所受的力和力矩
        :param motor_thrusts: 四个电机的推力 [f1, f2, f3, f4]
        """
        # 总推力
        total_thrust = np.sum(motor_thrusts)
        
        # 力矩
        moments = self.B @ motor_thrusts
        
        return total_thrust, moments[1], moments[2], moments[3]
    
    def step(self, motor_thrusts, dt=0.01):
        """
        执行一次动力学仿真步
        :param motor_thrusts: 四个电机推力
        :param dt: 时间步长
        """
        x, y, z, vx, vy, vz = self.state[:6]
        roll, pitch, yaw, p, q, r = self.state[6:]
        
        # 计算力和力矩
        thrust, Mx, My, Mz = self.compute_forces(motor_thrusts)
        
        # 旋转矩阵
        R = self.rotation_matrix()
        
        # 线加速度（世界坐标系）
        acc_world = (R @ np.array([0, 0, thrust]) / self.mass) - np.array([0, 0, self.g])
        
        # 角加速度
        I = np.diag([self.Ixx, self.Iyy, self.Izz])
        angular_acc = np.linalg.inv(I) @ (np.array([Mx, My, Mz]) - 
            np.cross(np.array([p, q, r]), I @ np.array([p, q, r])))
        
        # 更新状态（欧拉法）
        self.state[0] += vx * dt
        self.state[1] += vy * dt
        self.state[2] += vz * dt
        self.state[3] += acc_world[0] * dt
        self.state[4] += acc_world[1] * dt
        self.state[5] += acc_world[2] * dt
        self.state[6] += p * dt
        self.state[7] += q * dt
        self.state[8] += r * dt
        self.state[9] += angular_acc[0] * dt
        self.state[10] += angular_acc[1] * dt
        self.state[11] += angular_acc[2] * dt
    
    def hover_controller(self, target_z=2.0, kp_z=5.0, kd_z=2.0):
        """
        悬停控制器
        :param target_z: 目标高度
        :param kp_z: 高度比例系数
        :param kd_z: 高度微分系数
        """
        z = self.state[2]
        vz = self.state[5]
        
        # 计算所需总推力
        error_z = target_z - z
        thrust = self.mass * (self.g + kp_z * error_z + kd_z * (-vz))
        
        # 分配推力到四个电机（悬停时四个电机推力相等）
        motor_thrusts = np.array([thrust/4, thrust/4, thrust/4, thrust/4])
        
        return motor_thrusts

def simulate_hover():
    """模拟悬停控制"""
    drone = QuadrotorDynamics()
    drone.set_state(0, 0, 0)  # 从地面起飞
    
    dt = 0.01
    total_time = 10.0
    num_steps = int(total_time / dt)
    
    # 记录轨迹
    trajectory = []
    
    for _ in range(num_steps):
        # 获取控制输入
        motor_thrusts = drone.hover_controller(target_z=2.0)
        
        # 执行仿真步
        drone.step(motor_thrusts, dt)
        
        # 记录状态
        trajectory.append(drone.state[:3].copy())
    
    trajectory = np.array(trajectory)
    
    # 可视化
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(num_steps)*dt, trajectory[:, 2], 'b-', linewidth=2)
    ax.axhline(y=2.0, color='r', linestyle='--', label='目标高度')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('高度 (m)')
    ax.set_title('无人机悬停仿真')
    ax.legend()
    ax.grid(True)
    plt.savefig('hover_simulation.png')
    plt.close()
    
    print("悬停仿真完成！")
    print(f"最终高度: {trajectory[-1, 2]:.2f} m")

if __name__ == '__main__':
    simulate_hover()