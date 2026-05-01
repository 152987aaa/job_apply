"""
浙大高飞无人机团队 | A* 2D栅格全局路径规划
【严格对应链接步骤1】
功能：构建代价地图 → A*规划全局路径 → 输出路点（给PSO做初始种群）
教学重点：A*启发函数、代价地图、路径回溯
"""

import numpy as np
import heapq
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import save_waypoints, plot_path, calculate_path_length

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 全局参数（可调参学习）=====================
GRID_RESOLUTION = 1.0    # 栅格分辨率(米)
START_POINT = (2, 2)     # 无人机起点
GOAL_POINT = (22, 22)    # 任务终点
MAP_SHAPE = (25, 25)     # 2D地图尺寸

# ===================== 1. 构建代价地图（链接核心：激光点云→栅格）=====================
def build_costmap():
    """
    代价地图：0=可通行，1=障碍物
    模拟隧道/矿洞静态障碍物，对应链接退化场景
    """
    costmap = np.zeros(MAP_SHAPE)
    
    # 添加障碍物（模拟矿洞墙壁和障碍物）
    # 左侧墙壁
    costmap[5:9, 5:16] = 1
    # 右侧墙壁
    costmap[13:17, 7:21] = 1
    # 中间障碍物
    costmap[10:12, 12:14] = 1
    # 下方障碍物
    costmap[18:20, 5:10] = 1
    
    # 保存代价地图
    os.makedirs("../data/input", exist_ok=True)
    np.save("../data/input/costmap.npy", costmap)
    print("代价地图已构建并保存")
    
    return costmap

# ===================== 2. A*算法核心实现 =====================
class AStar:
    def __init__(self, costmap, start, goal):
        """
        初始化A*规划器
        :param costmap: 代价地图
        :param start: 起点坐标 (x, y)
        :param goal: 终点坐标 (x, y)
        """
        self.costmap = costmap
        self.start = start
        self.goal = goal
        self.map_x, self.map_y = costmap.shape
        
        # 8方向移动（无人机平面运动）
        self.motions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (1, -1), (-1, 1), (1, 1)]
        
        # 方向代价（直线移动代价为1，对角线为√2）
        self.motion_costs = [1.0, 1.0, 1.0, 1.0,
                             np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
    
    def heuristic(self, node):
        """
        启发函数：欧氏距离（A*灵魂，引导路径向终点）
        :param node: 当前节点坐标 (x, y)
        :return: 到终点的估计距离
        """
        return np.hypot(node[0] - self.goal[0], node[1] - self.goal[1])
    
    def is_valid(self, node):
        """
        检查节点是否有效（在地图范围内且无障碍物）
        :param node: 节点坐标 (x, y)
        :return: True=有效, False=无效
        """
        x, y = node
        if x < 0 or x >= self.map_x or y < 0 or y >= self.map_y:
            return False
        if self.costmap[x, y] == 1:
            return False
        return True
    
    def planning(self):
        """
        主规划函数：输出全局路径
        :return: 路径点列表 [(x1,y1), (x2,y2), ...]，失败返回None
        """
        # 优先队列：(总代价f, 当前代价g, 节点坐标, 父节点)
        open_heap = []
        heapq.heappush(open_heap, (self.heuristic(self.start), 0, self.start, None))
        
        # 记录已探索节点及其代价和父节点
        close_dict = {}  # key: node, value: (g_cost, parent_node)
        
        while open_heap:
            # 选取代价最小的节点（A*核心：总是扩展f值最小的节点）
            f_cost, g_cost, current_node, parent_node = heapq.heappop(open_heap)
            
            # 如果已探索过该节点且有更优路径，跳过
            if current_node in close_dict and close_dict[current_node][0] <= g_cost:
                continue
            
            # 记录当前节点（代价和父节点）
            close_dict[current_node] = (g_cost, parent_node)
            
            # 如果到达终点，回溯路径
            if current_node == self.goal:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    if current_node in close_dict:
                        current_node = close_dict[current_node][1]
                    else:
                        current_node = None
                # 反转路径（从起点到终点）
                path.reverse()
                print(f"A* planning completed! Path length: {len(path)} points")
                return path
            
            # 扩展8个方向
            for i, (dx, dy) in enumerate(self.motions):
                next_node = (current_node[0] + dx, current_node[1] + dy)
                
                # 检查有效性
                if not self.is_valid(next_node):
                    continue
                
                # 计算新的g代价
                new_g = g_cost + self.motion_costs[i]
                # 计算新的f代价
                new_f = new_g + self.heuristic(next_node)
                
                # 添加到优先队列
                heapq.heappush(open_heap, (new_f, new_g, next_node, current_node))
        
        # 如果队列为空仍未找到路径
        print("A*规划失败：无法找到可行路径")
        return None

# ===================== 3. 主函数 =====================
def main():
    print("=" * 60)
    print("A* 2D栅格全局路径规划")
    print("浙大高飞无人机团队")
    print("=" * 60)
    
    # 步骤1：构建代价地图
    print("\n【步骤1】构建代价地图")
    costmap = build_costmap()
    
    # 步骤2：初始化A*规划器
    print("\n【步骤2】初始化A*规划器")
    print(f"起点: {START_POINT}")
    print(f"终点: {GOAL_POINT}")
    print(f"地图尺寸: {MAP_SHAPE}")
    astar = AStar(costmap, START_POINT, GOAL_POINT)
    
    # 步骤3：执行路径规划
    print("\n【步骤3】执行A*路径规划")
    path = astar.planning()
    
    # 步骤4：保存结果
    if path:
        print("\n【步骤4】保存路径结果")
        # 转换为numpy数组
        path_array = np.array(path)
        
        # 保存路点（给PSO做初始种群）
        save_waypoints("../data/output/a_star_waypoints.txt", path_array)
        
        # 计算路径长度
        length = calculate_path_length(path)
        print(f"路径总长度: {length:.2f} 米")
        
        # 可视化1：基础路径图
        plot_path(costmap, path, "A*全局路径规划", "../data/output/a_star_path.png")
        
        # 可视化2：路径详细图（带网格）
        plt.figure(figsize=(10, 10))
        plt.imshow(costmap.T, cmap='gray_r', origin='lower', extent=[0, MAP_SHAPE[0], 0, MAP_SHAPE[1]])
        plt.plot([p[0] for p in path], [p[1] for p in path], 'r-', linewidth=2.5, label='规划路径')
        plt.plot([p[0] for p in path], [p[1] for p in path], 'ro', markersize=6)
        plt.scatter(path[0][0], path[0][1], c='green', s=150, marker='o', label='起点')
        plt.scatter(path[-1][0], path[-1][1], c='blue', s=150, marker='*', label='终点')
        plt.title('A*路径规划 - 详细视图', fontsize=14)
        plt.xlabel('X坐标', fontsize=12)
        plt.ylabel('Y坐标', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, color='white', linewidth=0.5)
        plt.savefig('../data/output/a_star_path_detail.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("详细路径图已保存")
        
        # 可视化3：路径长度分布图
        segment_lengths = []
        for i in range(len(path)-1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            segment_lengths.append(np.sqrt(dx**2 + dy**2))
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(segment_lengths)), segment_lengths, color='skyblue')
        plt.title('A*路径段长度分布', fontsize=14)
        plt.xlabel('路径段索引', fontsize=12)
        plt.ylabel('段长度 (米)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig('../data/output/a_star_segment_lengths.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("路径段长度分布图已保存")
        
        # 可视化4：创建路径动画（GIF）
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(costmap.T, cmap='gray_r', origin='lower', extent=[0, MAP_SHAPE[0], 0, MAP_SHAPE[1]])
        line, = ax.plot([], [], 'r-', linewidth=2.5)
        point, = ax.plot([], [], 'ro', markersize=10)
        ax.scatter(path[0][0], path[0][1], c='green', s=150, marker='o', label='起点')
        ax.scatter(path[-1][0], path[-1][1], c='blue', s=150, marker='*', label='终点')
        ax.set_title('A*路径搜索动画', fontsize=14)
        ax.legend()
        ax.grid(True, color='white', linewidth=0.5)
        
        def animate(i):
            line.set_data([p[0] for p in path[:i+1]], [p[1] for p in path[:i+1]])
            point.set_data([path[i][0]], [path[i][1]])
            return line, point
        
        anim = animation.FuncAnimation(fig, animate, frames=min(15, len(path)), interval=200, blit=True)
        anim.save('../data/output/a_star_animation.gif', writer='pillow', dpi=80)
        plt.close()
        print("路径搜索动画GIF已保存")
        
        print("\n[A* Done] Path planning completed!")
        print("Output files:")
        print("  - data/input/costmap.npy: 代价地图")
        print("  - data/output/a_star_waypoints.txt: 路点数据（供PSO使用）")
        print("  - data/output/a_star_path.png: 路径可视化")
        print("  - data/output/a_star_path_detail.png: 详细路径图")
        print("  - data/output/a_star_segment_lengths.png: 路径段长度分布")
        print("  - data/output/a_star_animation.gif: 路径搜索动画")
    else:
        print("\n[A* Failed] Path planning failed!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()