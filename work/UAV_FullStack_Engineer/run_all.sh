#!/bin/bash
# 浙大高飞无人机团队 | 全项目一键运行脚本
# 对齐知乎职业生涯规划与论文学习清单

echo "===== 浙大高飞无人机团队 - 全栈算法学习项目 ====="

# 创建必要的目录
mkdir -p logs

echo ""
echo "===== 阶段1: 基础学习 ====="

echo "1.1 ROS基础学习"
python 01_Base_Learning/ros_basic_demo.py 2>&1 | tee logs/ros_basic.log

echo ""
echo "1.2 代价地图构建"
python 01_Base_Learning/costmap_builder.py 2>&1 | tee logs/costmap.log

echo ""
echo "1.3 无人机动力学"
python 01_Base_Learning/uav_dynamics.py 2>&1 | tee logs/dynamics.log

echo ""
echo "===== 阶段2: 路径规划 ====="

echo "2.1 路径规划算法"
python 02_Path_Planning/path_planners.py 2>&1 | tee logs/planners.log

echo ""
echo "===== 阶段3: 协同规划 ====="

echo "3.1 全局-局部协同规划"
python 03_Cooperative_Planning/cooperative_planner.py 2>&1 | tee logs/cooperative.log

echo ""
echo "===== 阶段4: SLAM ====="

echo "4.1 SLAM模块"
python 04_SLAM/slam_module.py 2>&1 | tee logs/slam.log

echo ""
echo "===== 阶段5: 强化学习 ====="

echo "5.1 强化学习规划"
python 05_Reinforcement_Learning/rl_planner.py 2>&1 | tee logs/rl.log

echo ""
echo "===== 阶段6: 仿真验证 ====="

echo "6.1 仿真管理器"
python 06_Simulation/simulation_manager.py 2>&1 | tee logs/simulation.log

echo ""
echo "===== 阶段7: 嵌入式部署 ====="

echo "7.1 嵌入式模块"
python 07_Embedded_Deployment/embedded_module.py 2>&1 | tee logs/embedded.log

echo ""
echo "===== 阶段8: 行业场景 ====="

echo "8.1 行业场景演示"
python 08_Industry_Scenarios/industry_scenarios.py 2>&1 | tee logs/industry.log

echo ""
echo "===== 项目运行完成 ====="
echo "日志文件已保存至 logs/ 目录"
echo "生成的图片文件:"
ls -la *.png 2>/dev/null || echo "无图片文件生成"