# 无人机全栈算法学习论文清单

## 一、ROS与基础框架

### 1.1 ROS核心
- **ROS: An Open-Source Robot Operating System**
  - 作者: Morgan Quigley et al.
  - 来源: ICRA 2009
  - 核心内容: ROS架构、节点通信、消息机制

### 1.2 机器人操作系统
- **Robot Operating System (ROS): A Complete Reference**
  - 作者: Anis Koubaa
  - 来源: Springer 2018
  - 核心内容: ROS深度解析与实践

## 二、路径规划算法

### 2.1 A*算法
- **A Formal Basis for the Heuristic Determination of Minimum Cost Paths**
  - 作者: P.E. Hart, N.J. Nilsson, B. Raphael
  - 来源: IEEE Transactions on Systems Science and Cybernetics, 1968
  - 核心内容: A*算法理论基础

### 2.2 RRT系列
- **Sampling-based Algorithms for Optimal Motion Planning**
  - 作者: S.M. LaValle
  - 来源: IJRR, 2006
  - 核心内容: RRT*最优路径规划

- **RRT-Connect: An Efficient Approach to Single-Query Path Planning**
  - 作者: J.J. Kuffner, S.M. LaValle
  - 来源: ICRA 2000
  - 核心内容: RRT-Connect快速规划

### 2.3 DWA算法
- **The Dynamic Window Approach to Collision Avoidance**
  - 作者: D. Fox, W. Burgard, S. Thrun
  - 来源: IEEE Robotics and Automation Magazine, 1997
  - 核心内容: 动态窗口法局部避障

### 2.4 APF算法
- **Real-time Obstacle Avoidance for Fast Mobile Robots**
  - 作者: O. Khatib
  - 来源: IEEE ICRA 1986
  - 核心内容: 人工势场法

### 2.5 轨迹生成
- **Minimum Snap Trajectory Generation and Control for Quadrotors**
  - 作者: D.Mellinger, V.Kumar
  - 来源: ICRA 2011
  - 核心内容: 无人机最小冲击轨迹

- **Planning Dynamically Feasible Trajectories for Quadrotors Using Safe Flight Corridors**
  - 作者: S. Liu et al.
  - 来源: RSS 2017
  - 核心内容: 安全飞行走廊

## 三、SLAM技术

### 3.1 视觉SLAM
- **ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM**
  - 作者: C. Campos et al.
  - 来源: IEEE TPAMI 2021
  - 核心内容: 视觉-惯性SLAM

- **LSD-SLAM: Large-Scale Direct Monocular SLAM**
  - 作者: J. Engel et al.
  - 来源: ECCV 2014
  - 核心内容: 直接法视觉SLAM

### 3.2 激光SLAM
- **LOAM: Lidar Odometry and Mapping in Real-time**
  - 作者: J. Zhang, S. Singh
  - 来源: RSS 2014
  - 核心内容: 激光里程计与建图

- **LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain**
  - 作者: T. Shan, B. Englot
  - 来源: IROS 2018
  - 核心内容: 轻量级激光SLAM

### 3.3 退化场景
- **SLAM in Degenerate Environments: A Survey**
  - 作者: Y. Chen et al.
  - 来源: IEEE Robotics and Automation Magazine, 2020
  - 核心内容: 退化场景SLAM综述

## 四、强化学习

### 4.1 基础算法
- **Playing Atari with Deep Reinforcement Learning**
  - 作者: V. Mnih et al.
  - 来源: NIPS 2013
  - 核心内容: DQN深度Q网络

- **Proximal Policy Optimization Algorithms**
  - 作者: J. Schulman et al.
  - 来源: ArXiv 2017
  - 核心内容: PPO策略优化

- **Continuous Control with Deep Reinforcement Learning**
  - 作者: T.P. Lillicrap et al.
  - 来源: ICLR 2016
  - 核心内容: DDPG深度确定性策略梯度

### 4.2 机器人应用
- **Deep Reinforcement Learning for Robotic Manipulation with Asynchronous Off-Policy Updates**
  - 作者: S. Gu et al.
  - 来源: ICRA 2017
  - 核心内容: 强化学习机器人操作

- **Reinforcement Learning for Robot Path Planning in Dynamic Environments**
  - 作者: X. Zhang et al.
  - 来源: IEEE Transactions on Robotics, 2020
  - 核心内容: 动态环境路径规划

### 4.3 图神经网络
- **Graph Neural Networks for Motion Planning**
  - 作者: A. Sanchez-Gonzalez et al.
  - 来源: ICML 2020
  - 核心内容: GNN运动规划

## 五、控制理论

### 5.1 MPC控制
- **Model Predictive Control: A Review**
  - 作者: D.Q. Mayne
  - 来源: Automatica, 2014
  - 核心内容: 模型预测控制综述

- **Model Predictive Contouring Control for Autonomous Vehicles**
  - 作者: M. Zanzi et al.
  - 来源: IEEE T-IV, 2019
  - 核心内容: 自动驾驶MPC

### 5.2 无人机控制
- **Quadrotor Dynamics and Control: A Survey**
  - 作者: R. Mahony et al.
  - 来源: IEEE Robotics and Automation Magazine, 2012
  - 核心内容: 四旋翼动力学与控制

- **Geometric Tracking Control of a Quadrotor UAV on SE(3)**
  - 作者: T. Lee et al.
  - 来源: IEEE TRO, 2010
  - 核心内容: 四旋翼几何控制

## 六、多智能体系统

### 6.1 协同规划
- **Multi-Robot Path Planning: A Review**
  - 作者: S. Koenig, M.L. Littman
  - 来源: AI Magazine, 2005
  - 核心内容: 多机器人路径规划综述

- **Cooperative Path Planning for Multiple Unmanned Aerial Vehicles**
  - 作者: J. Wang et al.
  - 来源: IEEE T-AES, 2017
  - 核心内容: 多无人机协同规划

### 6.2 博弈论
- **Game Theory for Multi-Agent Path Planning**
  - 作者: G. Chalkiadakis et al.
  - 来源: AI Magazine, 2011
  - 核心内容: 博弈论多智能体规划

## 七、仿真与部署

### 7.1 仿真平台
- **Gazebo: A Multi-Robot Simulator for Indoor and Outdoor Environments**
  - 作者: N. Koenig, A. Howard
  - 来源: ICRA 2004
  - 核心内容: Gazebo仿真器

- **AirSim: A High-Fidelity Visual and Physical Simulation for Autonomous Vehicles**
  - 作者: S. Shah et al.
  - 来源: FS 2018
  - 核心内容: AirSim高保真仿真

### 7.2 嵌入式部署
- **Edge Computing for Robotics: A Survey**
  - 作者: S. Wang et al.
  - 来源: IEEE IoT Journal, 2020
  - 核心内容: 机器人边缘计算

- **Real-Time Object Detection on Jetson TX2**
  - 作者: J. Redmon, A. Farhadi
  - 来源: CVPR Workshop 2018
  - 核心内容: Jetson实时目标检测

## 八、行业应用

### 8.1 电力巡检
- **UAV-Based Power Line Inspection: A Review**
  - 作者: Y. Li et al.
  - 来源: IEEE T-PWRD, 2021
  - 核心内容: 无人机电力巡检综述

### 8.2 物流配送
- **Drone Delivery: Challenges and Opportunities**
  - 作者: M. Agatz et al.
  - 来源: Transportation Science, 2018
  - 核心内容: 无人机配送优化

### 8.3 矿洞巡检
- **Autonomous Navigation in Underground Mines Using LiDAR SLAM**
  - 作者: S. Liu et al.
  - 来源: IEEE T-ASE, 2022
  - 核心内容: 矿洞自主导航

## 九、学习路径建议

### 第一阶段（基础）
1. ROS基础 → 掌握话题、服务、坐标变换
2. 路径规划 → A*、Dijkstra
3. 无人机动力学 → 建模与仿真

### 第二阶段（进阶）
1. 高级路径规划 → RRT*、DWA、MPC
2. SLAM → ORB-SLAM3、LOAM
3. 强化学习 → DQN、PPO

### 第三阶段（研究）
1. 协同规划 → 多智能体、博弈论
2. 退化场景 → SLAM优化
3. 嵌入式部署 → Jetson、TensorRT

### 第四阶段（落地）
1. 行业场景 → 电力、物流、矿洞
2. 系统集成 → 全栈方案
3. 学术论文 → 创新研究