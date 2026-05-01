# 无人机路径规划工程

> 浙大高飞无人机团队 | A* → PSO → B-spline → Gazebo仿真

## 📁 项目结构

```
drone_path_planner/
├── a_star_global/          # A*全局路径规划
│   └── a_star_planner.py   # A*算法实现
├── pso_optimize/           # PSO路径优化
│   └── pso_optimizer.py    # PSO算法实现
├── bspline_smooth/         # B-spline轨迹平滑
│   └── bspline_generator.py # B-spline实现
├── gazebo_simulation/      # Gazebo仿真
│   ├── trajectory_tracker.py    # 轨迹跟踪控制器
│   └── trajectory_tracking.launch # 启动脚本
├── env_setup/              # 环境配置
│   └── install_dependencies.sh  # 安装脚本
├── utils/                  # 通用工具
│   ├── __init__.py
│   └── common.py           # 工具函数
├── docs/                   # 教学文档
│   └── teach.md            # 教学资料
├── data/                   # 数据文件夹
│   ├── input/              # 输入数据
│   └── output/             # 输出数据
├── main.py                 # 主程序
└── README.md               # 项目说明
```

## 🚀 快速开始

### 1. 运行完整流程

```bash
cd drone_path_planner
python main.py
```

### 2. 分步运行

```bash
# 步骤1：A*路径规划
python a_star_global/a_star_planner.py

# 步骤2：PSO路径优化
python pso_optimize/pso_optimizer.py

# 步骤3：B-spline轨迹平滑
python bspline_smooth/bspline_generator.py

# 步骤4：Gazebo仿真（需要ROS环境）
roslaunch gazebo_simulation trajectory_tracking.launch
```

### 3. 环境安装

```bash
# Ubuntu 20.04 + ROS Noetic
sudo chmod +x env_setup/install_dependencies.sh
sudo ./env_setup/install_dependencies.sh
```

## 📊 算法流程

```
代价地图 → A*规划 → 离散路点 → PSO优化 → 优化路点 → B-spline平滑 → 连续轨迹 → Gazebo跟踪
```

## 🎯 学习路径

| 模块 | 学习目标 | 核心知识点 |
|------|----------|------------|
| A* | 全局路径搜索 | 启发式搜索、优先队列、路径回溯 |
| PSO | 路径优化 | 群智能算法、惯性权重、认知/社会系数 |
| B-spline | 轨迹生成 | 参数化方法、导数计算、平滑约束 |
| Gazebo | 仿真验证 | ROS消息、PID控制、无人机模型 |

## 📝 输出文件

```
data/output/
├── a_star_waypoints.txt      # A*路点
├── a_star_path.png           # A*路径图
├── pso_optimized_waypoints.txt # PSO优化路点
├── pso_convergence.png       # PSO收敛曲线
├── bspline_trajectory.txt    # B-spline轨迹
├── bspline_trajectory.png    # 轨迹对比图
├── trajectory_velocity.txt   # 速度数据
└── trajectory_acceleration.txt # 加速度数据
```

## 📚 参考资料

1. *Probabilistic Robotics* - Sebastian Thrun
2. rotors_simulator: https://github.com/ethz-asl/rotors_simulator
3. ROS官方文档: http://wiki.ros.org/

## 📧 联系方式

浙大高飞无人机团队