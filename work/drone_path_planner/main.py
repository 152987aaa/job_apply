"""
浙大高飞无人机团队 | 无人机路径规划主程序
【完整流程】A* → PSO → B-spline → 仿真
"""

import os
import sys

def run_a_star():
    """运行A*路径规划"""
    print("\n" + "=" * 60)
    print("【步骤1】运行A*全局路径规划")
    print("=" * 60)
    os.system("python a_star_global/a_star_planner.py")

def run_pso():
    """运行PSO路径优化"""
    print("\n" + "=" * 60)
    print("【步骤2】运行PSO路径优化")
    print("=" * 60)
    os.system("python pso_optimize/pso_optimizer.py")

def run_bspline():
    """运行B-spline轨迹平滑"""
    print("\n" + "=" * 60)
    print("【步骤3】运行B-spline轨迹平滑")
    print("=" * 60)
    os.system("python bspline_smooth/bspline_generator.py")

def show_results():
    """显示结果汇总"""
    print("\n" + "=" * 60)
    print("【结果汇总】")
    print("=" * 60)
    
    # 读取结果文件
    try:
        import numpy as np
        
        # A*路径长度
        a_star_path = np.loadtxt("data/output/a_star_waypoints.txt")
        a_star_length = 0
        for i in range(len(a_star_path)-1):
            a_star_length += np.hypot(a_star_path[i+1][0]-a_star_path[i][0],
                                     a_star_path[i+1][1]-a_star_path[i][1])
        
        # PSO优化后路径长度
        pso_path = np.loadtxt("data/output/pso_optimized_waypoints.txt")
        pso_length = 0
        for i in range(len(pso_path)-1):
            pso_length += np.hypot(pso_path[i+1][0]-pso_path[i][0],
                                  pso_path[i+1][1]-pso_path[i][1])
        
        # B-spline轨迹长度
        bspline_traj = np.loadtxt("data/output/bspline_trajectory.txt")
        bspline_length = 0
        for i in range(len(bspline_traj)-1):
            bspline_length += np.hypot(bspline_traj[i+1][0]-bspline_traj[i][0],
                                      bspline_traj[i+1][1]-bspline_traj[i][1])
        
        print(f"A*路径长度: {a_star_length:.2f} 米")
        print(f"PSO优化后路径长度: {pso_length:.2f} 米")
        print(f"B-spline平滑轨迹长度: {bspline_length:.2f} 米")
        print(f"PSO优化率: {((a_star_length - pso_length)/a_star_length*100):.2f}%")
        
    except Exception as e:
        print(f"读取结果失败: {e}")
    
    print("\n生成的文件：")
    print("  ├── data/input/costmap.npy")
    print("  ├── data/output/a_star_waypoints.txt")
    print("  ├── data/output/a_star_path.png")
    print("  ├── data/output/pso_optimized_waypoints.txt")
    print("  ├── data/output/pso_convergence.png")
    print("  ├── data/output/bspline_trajectory.txt")
    print("  ├── data/output/bspline_trajectory.png")
    print("  ├── data/output/trajectory_velocity.txt")
    print("  └── data/output/trajectory_acceleration.txt")

def main():
    """主函数"""
    print("=" * 60)
    print("无人机路径规划完整流程")
    print("浙大高飞无人机团队")
    print("=" * 60)
    print("\n流程：A*全局规划 → PSO路径优化 → B-spline轨迹平滑")
    
    # 步骤1：A*路径规划
    run_a_star()
    
    # 步骤2：PSO路径优化
    run_pso()
    
    # 步骤3：B-spline轨迹平滑
    run_bspline()
    
    # 显示结果
    show_results()
    
    print("\n" + "=" * 60)
    print("路径规划流程完成！")
    print("下一步：在Gazebo中运行无人机轨迹跟踪")
    print("命令：roslaunch gazebo_simulation trajectory_tracking.launch")
    print("=" * 60)

if __name__ == "__main__":
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()