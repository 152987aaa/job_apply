#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浙大高飞无人机团队 | 无人机全栈算法工程师学习项目
对应知乎职业生涯规划与论文学习清单
"""

import os
import sys

def run_module(module_name):
    """运行指定模块"""
    print(f"\n{'='*60}")
    print(f"运行模块: {module_name}")
    print(f"{'='*60}")
    
    module_paths = {
        'ros': '01_Base_Learning/ros_basic_demo.py',
        'costmap': '01_Base_Learning/costmap_builder.py',
        'dynamics': '01_Base_Learning/uav_dynamics.py',
        'planners': '02_Path_Planning/path_planners.py',
        'cooperative': '03_Cooperative_Planning/cooperative_planner.py',
        'slam': '04_SLAM/slam_module.py',
        'rl': '05_Reinforcement_Learning/rl_planner.py',
        'simulation': '06_Simulation/simulation_manager.py',
        'embedded': '07_Embedded_Deployment/embedded_module.py',
        'industry': '08_Industry_Scenarios/industry_scenarios.py'
    }
    
    if module_name in module_paths:
        script_path = module_paths[module_name]
        if os.path.exists(script_path):
            os.system(f'python {script_path}')
        else:
            print(f"错误：模块文件不存在: {script_path}")
    else:
        print(f"错误：未知模块: {module_name}")

def show_modules():
    """显示可用模块"""
    print("可用模块列表:")
    modules = [
        ('ros', 'ROS基础学习'),
        ('costmap', '代价地图构建'),
        ('dynamics', '无人机动力学'),
        ('planners', '路径规划算法'),
        ('cooperative', '协同规划'),
        ('slam', 'SLAM模块'),
        ('rl', '强化学习'),
        ('simulation', '仿真验证'),
        ('embedded', '嵌入式部署'),
        ('industry', '行业场景')
    ]
    
    for key, desc in modules:
        print(f"  {key:15} - {desc}")

def show_structure():
    """显示项目结构"""
    print("项目结构:")
    structure = """
UAV_FullStack_Engineer/
├── 01_Base_Learning/          # 基础学习
│   ├── ros_basic_demo.py      # ROS基础
│   ├── costmap_builder.py     # 代价地图
│   └── uav_dynamics.py        # 无人机动力学
├── 02_Path_Planning/          # 路径规划
│   └── path_planners.py       # A*, RRT*, DWA, APF
├── 03_Cooperative_Planning/   # 协同规划
│   └── cooperative_planner.py # 全局+局部协同
├── 04_SLAM/                   # SLAM模块
│   └── slam_module.py         # 激光SLAM+粒子滤波
├── 05_Reinforcement_Learning/ # 强化学习
│   └── rl_planner.py          # DQN, PPO
├── 06_Simulation/             # 仿真验证
│   └── simulation_manager.py  # Gazebo+PX4 SITL
├── 07_Embedded_Deployment/    # 嵌入式部署
│   └── embedded_module.py     # Jetson+Pixhawk
├── 08_Industry_Scenarios/     # 行业场景
│   └── industry_scenarios.py  # 电力/物流/矿洞
├── 09_Publications/           # 学术论文
│   └── paper_list.md          # 论文清单
└── main.py                    # 主程序
    """
    print(structure)

def main():
    """主函数"""
    print("="*60)
    print("浙大高飞无人机团队 | 无人机全栈算法工程师学习项目")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  python main.py <模块名>")
        print("\n示例:")
        print("  python main.py planners    # 运行路径规划模块")
        print("  python main.py slam        # 运行SLAM模块")
        print("\n输入 'list' 查看所有模块")
        print("输入 'structure' 查看项目结构")
        return
    
    command = sys.argv[1]
    
    if command == 'list':
        show_modules()
    elif command == 'structure':
        show_structure()
    else:
        run_module(command)

if __name__ == '__main__':
    main()