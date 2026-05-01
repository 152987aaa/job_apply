#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浙大高飞无人机团队 | SLAM模块
教学目标：掌握激光SLAM和视觉SLAM基础
对应论文：
- ORB-SLAM3: 《ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM》
- LOAM: 《LOAM: Lidar Odometry and Mapping in Real-time》
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

class LaserOdometry:
    """激光里程计"""
    
    def __init__(self):
        self.prev_points = None
        self.prev_pose = np.eye(4)
    
    def icp(self, points1, points2, max_iter=50, tolerance=1e-6):
        """
        迭代最近点(ICP)算法
        :param points1: 参考点云
        :param points2: 当前点云
        :param max_iter: 最大迭代次数
        :param tolerance: 收敛阈值
        :return: 变换矩阵
        """
        R = np.eye(3)
        t = np.zeros(3)
        
        for _ in range(max_iter):
            # 寻找最近点
            tree = KDTree(points1)
            distances, indices = tree.query(points2, k=1)
            correspondences = points1[indices.flatten()]
            
            # 计算质心
            centroid1 = np.mean(correspondences, axis=0)
            centroid2 = np.mean(points2, axis=0)
            
            # 去质心
            points1_centered = correspondences - centroid1
            points2_centered = points2 - centroid2
            
            # 计算H矩阵
            H = points2_centered.T @ points1_centered
            
            # SVD分解
            U, S, Vt = np.linalg.svd(H)
            R_new = U @ Vt
            
            # 确保旋转矩阵行列式为1
            if np.linalg.det(R_new) < 0:
                Vt[-1, :] *= -1
                R_new = U @ Vt
            
            # 计算平移
            t_new = centroid1 - R_new @ centroid2
            
            # 更新变换
            R = R_new @ R
            t = R_new @ t + t_new
            
            # 检查收敛
            mean_dist = np.mean(distances)
            if mean_dist < tolerance:
                break
        
        # 构建变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
    
    def update(self, current_points):
        """
        更新里程计
        :param current_points: 当前激光点云
        :return: 当前位姿
        """
        if self.prev_points is None:
            self.prev_points = current_points
            return self.prev_pose
        
        # 执行ICP
        T = self.icp(self.prev_points, current_points)
        
        # 更新位姿
        self.prev_pose = self.prev_pose @ T
        self.prev_points = current_points
        
        return self.prev_pose

class ParticleFilterSLAM:
    """粒子滤波SLAM"""
    
    def __init__(self, num_particles=100):
        """
        初始化粒子滤波器
        :param num_particles: 粒子数量
        """
        self.num_particles = num_particles
        self.particles = []
        self.weights = np.ones(num_particles) / num_particles
        
        # 初始化粒子
        for _ in range(num_particles):
            self.particles.append({
                'pose': np.eye(4),
                'map': []
            })
    
    def predict(self, odometry):
        """
        预测步骤
        :param odometry: 里程计测量
        """
        for i in range(self.num_particles):
            # 添加噪声
            noise = np.random.normal(0, 0.01, (4, 4))
            noise[3, :] = 0
            noise[:, 3] = 0
            noise[3, 3] = 1
            
            self.particles[i]['pose'] = self.particles[i]['pose'] @ odometry @ (np.eye(4) + noise)
    
    def update(self, landmarks):
        """
        更新步骤
        :param landmarks: 观测到的路标
        """
        for i in range(self.num_particles):
            # 计算观测似然
            likelihood = 1.0
            for landmark in landmarks:
                # 预测观测
                predicted_pos = self.particles[i]['pose'] @ np.array([landmark[0], landmark[1], 0, 1])
                
                # 计算距离
                dist = np.hypot(predicted_pos[0] - landmark[2], predicted_pos[1] - landmark[3])
                likelihood *= np.exp(-dist**2 / (2 * 0.1**2))
            
            self.weights[i] *= likelihood
        
        # 归一化权重
        self.weights /= np.sum(self.weights)
    
    def resample(self):
        """重采样步骤"""
        # 系统重采样
        indices = np.random.choice(range(self.num_particles), 
                                   size=self.num_particles, 
                                   p=self.weights)
        
        new_particles = []
        new_weights = np.ones(self.num_particles) / self.num_particles
        
        for idx in indices:
            new_particles.append({
                'pose': self.particles[idx]['pose'].copy(),
                'map': list(self.particles[idx]['map'])
            })
        
        self.particles = new_particles
        self.weights = new_weights
    
    def get_best_pose(self):
        """获取最优位姿"""
        # 选择权重最大的粒子
        best_idx = np.argmax(self.weights)
        return self.particles[best_idx]['pose']

class DegenerateEnvironmentHandler:
    """退化场景处理"""
    
    def __init__(self):
        self.feature_tracker = []
    
    def detect_degeneration(self, points):
        """
        检测退化场景
        :param points: 激光点云
        :return: 是否退化
        """
        # 检查点云分布
        cov_matrix = np.cov(points.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        
        # 如果特征值差异很大，说明可能处于退化场景
        ratio = np.max(eigenvalues) / np.min(eigenvalues)
        return ratio > 100
    
    def enhance_features(self, points):
        """
        增强特征提取
        :param points: 原始点云
        :return: 增强后的特征点
        """
        # 提取边缘点
        edges = self.extract_edges(points)
        
        # 提取平面点
        planes = self.extract_planes(points)
        
        return {'edges': edges, 'planes': planes}
    
    def extract_edges(self, points, threshold=0.1):
        """提取边缘特征点"""
        edges = []
        for i in range(len(points)):
            # 计算邻域点的法向量变化
            neighbors = points[np.linalg.norm(points - points[i], axis=1) < 0.5]
            if len(neighbors) > 5:
                cov = np.cov(neighbors.T)
                eigvals = np.linalg.eigvals(cov)
                if np.min(eigvals) / np.max(eigvals) < threshold:
                    edges.append(points[i])
        return np.array(edges)
    
    def extract_planes(self, points, threshold=0.05):
        """提取平面特征点"""
        planes = []
        for i in range(len(points)):
            neighbors = points[np.linalg.norm(points - points[i], axis=1) < 0.5]
            if len(neighbors) > 10:
                # 拟合平面
                centroid = np.mean(neighbors, axis=0)
                centered = neighbors - centroid
                cov = centered.T @ centered
                eigvals, eigvecs = np.linalg.eig(cov)
                normal = eigvecs[:, np.argmin(eigvals)]
                
                # 检查平面拟合质量
                distances = np.abs(centered @ normal)
                if np.mean(distances) < threshold:
                    planes.append(points[i])
        return np.array(planes)

def test_slam():
    """测试SLAM模块"""
    print("测试激光里程计...")
    laser_odom = LaserOdometry()
    
    # 生成模拟点云
    points1 = np.random.rand(100, 3) * 10
    points2 = points1 + np.array([1.0, 0.5, 0.0])
    
    # 更新里程计
    pose = laser_odom.update(points1)
    pose = laser_odom.update(points2)
    print(f"里程计位姿:\n{pose}")
    
    print("\n测试粒子滤波SLAM...")
    pf_slam = ParticleFilterSLAM(num_particles=50)
    
    # 模拟里程计数据
    odom = np.eye(4)
    odom[0, 3] = 0.5
    odom[1, 3] = 0.3
    
    pf_slam.predict(odom)
    pf_slam.update([(1.0, 1.0, 1.5, 1.3)])
    pf_slam.resample()
    
    best_pose = pf_slam.get_best_pose()
    print(f"最优位姿:\n{best_pose}")
    
    print("\n测试退化场景处理...")
    deg_handler = DegenerateEnvironmentHandler()
    
    # 生成退化场景点云（一维分布）
    deg_points = np.zeros((100, 3))
    deg_points[:, 0] = np.linspace(0, 10, 100)
    
    is_degenerate = deg_handler.detect_degeneration(deg_points)
    print(f"是否退化场景: {is_degenerate}")
    
    print("\nSLAM模块测试完成！")

if __name__ == '__main__':
    test_slam()