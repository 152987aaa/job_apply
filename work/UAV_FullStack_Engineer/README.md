# 无人机全栈算法工程师学习项目

> 浙大高飞无人机团队 | 对齐知乎职业生涯规划与论文学习清单

## 📁 项目结构

```
UAV_FullStack_Engineer/
├── 01_Base_Learning/          # 基础学习模块
│   ├── ros_basic_demo.py      # ROS话题/服务/坐标变换
│   ├── costmap_builder.py     # 激光点云→代价地图
│   └── uav_dynamics.py        # 无人机动力学仿真
├── 02_Path_Planning/          # 路径规划模块
│   └── path_planners.py       # A*/RRT*/DWA/APF/Dijkstra/Dubins
├── 03_Cooperative_Planning/   # 协同规划模块
│   └── cooperative_planner.py # 全局+局部协同+MPC
├── 04_SLAM/                   # SLAM模块
│   └── slam_module.py         # 激光里程计/粒子滤波/退化场景处理
├── 05_Reinforcement_Learning/ # 强化学习模块
│   └── rl_planner.py          # DQN/PPO路径规划
├── 06_Simulation/             # 仿真验证模块
│   └── simulation_manager.py  # Gazebo/PX4 SITL仿真
├── 07_Embedded_Deployment/    # 嵌入式部署模块
│   └── embedded_module.py     # Jetson/Pixhawk部署
├── 08_Industry_Scenarios/     # 行业场景模块
│   └── industry_scenarios.py  # 电力巡检/物流配送/矿洞巡检
├── 09_Publications/           # 学术论文模块
│   └── paper_list.md          # 论文学习清单
├── main.py                    # 主程序入口
├── run_all.sh                 # 一键运行脚本
└── README.md                  # 项目说明
```

## 🚀 学习路径

| 阶段 | 模块 | 核心内容 | 对应论文 |
|------|------|----------|----------|
| 1 | 基础学习 | ROS、代价地图、动力学 | ROS论文、UAV Dynamics |
| 2 | 路径规划 | A*、RRT*、DWA、APF | A*论文、RRT*论文、DWA论文 |
| 3 | 协同规划 | 全局+局部协同、MPC | Global-Local Planning |
| 4 | SLAM | 激光SLAM、退化场景 | LOAM、ORB-SLAM3 |
| 5 | 强化学习 | DQN、PPO、GNN | PPO论文、DQN论文 |
| 6 | 仿真验证 | Gazebo、PX4 SITL | Gazebo论文 |
| 7 | 嵌入式部署 | Jetson、TensorRT | Jetson文档 |
| 8 | 行业场景 | 电力、物流、矿洞 | 行业技术规范 |

## 📖 运行方式

### 1. 运行单个模块

```bash
# 查看可用模块
python main.py list

# 运行路径规划模块
python main.py planners

# 运行SLAM模块
python main.py slam

# 运行行业场景模块
python main.py industry
```

### 2. 一键运行所有模块

```bash
bash run_all.sh
```

### 3. 分步运行

```bash
# 基础学习
python 01_Base_Learning/costmap_builder.py

# 路径规划
python 02_Path_Planning/path_planners.py

# SLAM
python 04_SLAM/slam_module.py

# 强化学习
python 05_Reinforcement_Learning/rl_planner.py
```

## 📊 核心算法清单

### 路径规划
| 算法 | 类型 | 适用场景 |
|------|------|----------|
| A* | 全局最优 | 静态环境 |
| RRT* | 采样最优 | 高维空间 |
| DWA | 局部避障 | 动态环境 |
| APF | 人工势场 | 实时避障 |
| Dijkstra | 全局最优 | 权重图 |
| Dubins | 轨迹生成 | 无人机路径 |

### SLAM
| 算法 | 类型 | 适用场景 |
|------|------|----------|
| LOAM | 激光SLAM | 室外场景 |
| ORB-SLAM3 | 视觉SLAM | 室内外 |
| Particle Filter | 定位 | 多传感器融合 |

### 强化学习
| 算法 | 类型 | 适用场景 |
|------|------|----------|
| DQN | 值函数 | 离散动作 |
| PPO | 策略优化 | 连续动作 |
| DDPG | 策略梯度 | 连续控制 |

## 🎯 能力目标

### 基础阶段
- ✅ 掌握ROS1框架
- ✅ 理解代价地图构建
- ✅ 掌握无人机动力学

### 进阶阶段
- ✅ 实现多种路径规划算法
- ✅ 掌握全局-局部协同
- ✅ 理解SLAM原理

### 研究阶段
- ✅ 掌握强化学习方法
- ✅ 理解仿真验证流程
- ✅ 掌握嵌入式部署

### 落地阶段
- ✅ 适配行业场景
- ✅ 系统集成能力
- ✅ 学术论文阅读

## 📚 论文清单

项目配套论文清单见 `09_Publications/paper_list.md`，包含：
- ROS与基础框架 (5篇)
- 路径规划算法 (10篇)
- SLAM技术 (8篇)
- 强化学习 (6篇)
- 控制理论 (5篇)
- 多智能体系统 (4篇)
- 仿真与部署 (4篇)
- 行业应用 (3篇)

## 🛠️ 环境要求

```bash
# Python依赖
pip install numpy matplotlib scipy tensorflow onnxruntime

# ROS依赖（可选，用于仿真）
# ROS Noetic + Gazebo11 + rotors_simulator

# 嵌入式依赖（可选）
# Jetson Orin + TensorRT + Pixhawk
```

## 📧 联系方式

浙大高飞无人机团队

---

**学习建议**: 按照项目结构从基础到进阶逐步学习，配合论文清单深入理解算法原理，通过仿真验证和实际部署巩固知识。