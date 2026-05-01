#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A*算法路径规划 - 浙江大学高飞无人机团队教学材料

本文件实现了经典的A*路径搜索算法，用于在2D网格地图上寻找最优路径。
这是无人机自主导航的基础算法，理解此算法是学习无人机路径规划的第一步。

教学目标：
1. 理解A*算法的核心原理（启发式搜索）
2. 掌握open/closed列表的管理机制
3. 理解启发函数的设计原则
4. 学会在网格地图上应用路径规划算法
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq  # 用于实现优先队列

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 第一步：定义地图和常量
# ==============================================================================

class GridMap:
    """
    网格地图类 - 用于表示无人机飞行环境
    
    属性：
        grid: 二维数组，表示地图，0=可通行，1=障碍物
        width: 地图宽度（列数）
        height: 地图高度（行数）
    """
    
    def __init__(self, width=20, height=15):
        """
        初始化地图
        
        参数：
            width: 地图宽度，默认20列
            height: 地图高度，默认15行
        """
        self.width = width
        self.height = height
        # 创建空地图，初始全为可通行区域(0)
        self.grid = np.zeros((height, width), dtype=int)
    
    def add_obstacle(self, x, y):
        """
        在地图上添加障碍物
        
        参数：
            x: 障碍物的x坐标（列）
            y: 障碍物的y坐标（行）
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1  # 1表示障碍物
    
    def add_obstacle_rect(self, x1, y1, x2, y2):
        """
        添加矩形障碍物区域
        
        参数：
            x1, y1: 矩形左上角坐标
            x2, y2: 矩形右下角坐标
        """
        for y in range(min(y1, y2), max(y1, y2) + 1):
            for x in range(min(x1, x2), max(x1, x2) + 1):
                self.add_obstacle(x, y)
    
    def is_passable(self, x, y):
        """
        检查某个位置是否可通行
        
        参数：
            x: 检查位置的x坐标
            y: 检查位置的y坐标
        
        返回：
            True: 可通行
            False: 不可通行（障碍物或边界外）
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False  # 边界外不可通行
        return self.grid[y, x] == 0  # 0表示可通行
    
    def plot(self, path=None, start=None, goal=None, visited_nodes=None):
        """
        可视化地图和路径
        
        参数：
            path: 路径点列表，每个点为(x, y)元组
            start: 起点坐标
            goal: 终点坐标
            visited_nodes: 访问过的节点列表（用于动画）
        """
        plt.figure(figsize=(10, 7))
        plt.imshow(self.grid, cmap='Greys', origin='lower')
        
        if visited_nodes:
            visited_x = [node[0] for node in visited_nodes]
            visited_y = [node[1] for node in visited_nodes]
            plt.scatter(visited_x, visited_y, c='yellow', s=20, alpha=0.5, label='搜索过的节点')
        
        if path:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            plt.plot(path_x, path_y, 'r-', linewidth=2.5, label='路径')
            plt.scatter(path_x, path_y, c='red', s=60, zorder=5)
        
        if start:
            plt.scatter(start[0], start[1], c='green', s=150, marker='o', zorder=6, label='起点')
        
        if goal:
            plt.scatter(goal[0], goal[1], c='blue', s=150, marker='*', zorder=6, label='终点')
        
        plt.title('A*路径规划结果', fontsize=14)
        plt.xlabel('X坐标', fontsize=12)
        plt.ylabel('Y坐标', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', color='black', linewidth=0.5)
        plt.savefig('./AStar路径规划.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("路径规划图已保存为 'AStar路径规划.png'")
        
        # 可视化2：路径详细图（带网格标注）
        plt.figure(figsize=(10, 7))
        plt.imshow(self.grid, cmap='Greys', origin='lower')
        if path:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            plt.plot(path_x, path_y, 'r-', linewidth=2.5)
            for i, (x, y) in enumerate(path):
                plt.text(x, y, str(i), color='white', fontsize=8, ha='center', va='center')
        plt.title('A*路径规划 - 路点序号标注', fontsize=14)
        plt.grid(True, which='both', color='black', linewidth=0.5)
        plt.savefig('./AStar路径规划_标注.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("路点标注图已保存为 'AStar路径规划_标注.png'")
        
        # 可视化3：创建搜索动画（GIF）
        if path and visited_nodes:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(self.grid, cmap='Greys', origin='lower')
            visited_scatter = ax.scatter([], [], c='yellow', s=20, alpha=0.5, label='搜索过的节点')
            path_line, = ax.plot([], [], 'r-', linewidth=2.5, label='路径')
            path_point, = ax.plot([], [], 'ro', markersize=8)
            ax.scatter(start[0], start[1], c='green', s=150, marker='o', label='起点')
            ax.scatter(goal[0], goal[1], c='blue', s=150, marker='*', label='终点')
            ax.set_title('A*搜索过程动画', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, which='both', color='black', linewidth=0.5)
            
            visited_history = []
            path_index = 0
            
            def animate_search(i):
                nonlocal path_index
                if i < len(visited_nodes):
                    visited_history.append(visited_nodes[i])
                    visited_scatter.set_offsets(visited_history)
                    if visited_nodes[i] in path:
                        idx = path.index(visited_nodes[i])
                        path_line.set_data([p[0] for p in path[:idx+1]], [p[1] for p in path[:idx+1]])
                        path_point.set_data([path[idx][0]], [path[idx][1]])
                return visited_scatter, path_line, path_point
            
            anim = animation.FuncAnimation(fig, animate_search, frames=min(100, len(visited_nodes)), 
                                           interval=50, blit=True)
            anim.save('./AStar搜索动画.gif', writer='pillow', dpi=100)
            plt.close()
            print("搜索动画已保存为 'AStar搜索动画.gif'")

# ==============================================================================
# 第二步：定义A*算法的节点类
# ==============================================================================

class Node:
    """
    A*算法的节点类
    
    A*算法中的每个节点代表搜索过程中的一个位置，存储：
    - 当前位置坐标
    - 从起点到当前节点的代价(g值)
    - 从当前节点到终点的估计代价(h值)
    - 总代价(f = g + h)
    - 父节点指针（用于回溯路径）
    """
    
    def __init__(self, x, y, g=0, h=0, parent=None):
        """
        初始化节点
        
        参数：
            x: 当前节点的x坐标
            y: 当前节点的y坐标
            g: 从起点到当前节点的实际代价
            h: 从当前节点到终点的估计代价
            parent: 父节点指针
        """
        self.x = x
        self.y = y
        self.g = g  # 起点到当前节点的代价
        self.h = h  # 启发式估计（当前节点到终点的代价）
        self.f = g + h  # 总代价
        self.parent = parent  # 父节点，用于回溯路径
    
    def __lt__(self, other):
        """
        定义节点比较规则，用于优先队列排序
        
        A*算法总是选择f值最小的节点进行扩展，所以需要比较f值
        """
        return self.f < other.f

# ==============================================================================
# 第三步：实现A*算法核心逻辑
# ==============================================================================

class AStarPlanner:
    """
    A*路径规划器
    
    核心算法流程：
    1. 将起点加入open列表
    2. 从open列表中选择f值最小的节点
    3. 将该节点从open列表移到closed列表
    4. 扩展该节点的所有邻居
    5. 如果邻居不在closed列表且不在open列表，或有更优路径，则加入/更新open列表
    6. 重复步骤2-5，直到找到终点或open列表为空
    """
    
    def __init__(self, grid_map):
        """
        初始化规划器
        
        参数：
            grid_map: GridMap对象，包含地图信息
        """
        self.grid_map = grid_map
        # 定义8个方向的移动：上下左右 + 四个对角
        # 每个方向用(dx, dy)表示坐标变化
        self.directions = [
            (0, 1),   # 上
            (0, -1),  # 下
            (1, 0),   # 右
            (-1, 0),  # 左
            (1, 1),   # 右上
            (1, -1),  # 右下
            (-1, 1),  # 左上
            (-1, -1)  # 左下
        ]
    
    def heuristic(self, x1, y1, x2, y2):
        """
        启发函数：估计从(x1,y1)到(x2,y2)的代价
        
        常用的启发函数有：
        1. 曼哈顿距离（Manhattan Distance）：|x1-x2| + |y1-y2|
           - 适用于只能沿网格线移动的情况（4方向移动）
        2. 欧几里得距离（Euclidean Distance）：sqrt((x1-x2)^2 + (y1-y2)^2)
           - 适用于可以对角线移动的情况（8方向移动）
        
        这里我们使用欧几里得距离，因为允许对角线移动
        """
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def get_neighbors(self, x, y):
        """
        获取某个节点的所有有效邻居
        
        参数：
            x: 当前节点的x坐标
            y: 当前节点的y坐标
        
        返回：
            邻居坐标列表，每个元素为(x, y)元组
        """
        neighbors = []
        for dx, dy in self.directions:
            new_x = x + dx
            new_y = y + dy
            # 检查邻居是否在地图范围内且可通行
            if self.grid_map.is_passable(new_x, new_y):
                neighbors.append((new_x, new_y))
        return neighbors
    
    def calculate_cost(self, x1, y1, x2, y2):
        """
        计算从一个节点移动到另一个节点的代价
        
        参数：
            x1, y1: 起始节点坐标
            x2, y2: 目标节点坐标
        
        返回：
            移动代价（直线移动为1，对角线移动为sqrt(2)）
        """
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if dx == 1 and dy == 1:
            return np.sqrt(2)  # 对角线移动代价
        else:
            return 1.0  # 直线移动代价
    
    def plan(self, start, goal):
        """
        A*路径规划主函数
        
        参数：
            start: 起点坐标，(x, y)元组
            goal: 终点坐标，(x, y)元组
        
        返回：
            path: 路径点列表，从起点到终点的坐标序列
                  如果无法找到路径，返回None
        """
        print(f"开始A*路径规划，起点: {start}，终点: {goal}")
        
        # 初始化open列表和closed列表
        open_list = []
        closed_list = set()  # 使用集合存储已访问的节点
        visited_nodes = []  # 记录访问过的节点（用于可视化）
        
        # 创建起点节点并加入open列表
        start_node = Node(start[0], start[1], g=0, h=self.heuristic(start[0], start[1], goal[0], goal[1]))
        heapq.heappush(open_list, start_node)
        
        # 记录节点信息，用于检测更优路径
        # key: (x, y)，value: Node对象
        node_info = {(start[0], start[1]): start_node}
        
        iteration = 0  # 迭代计数器，用于调试
        
        while open_list:
            iteration += 1
            
            # 从open列表中取出f值最小的节点
            current_node = heapq.heappop(open_list)
            current_pos = (current_node.x, current_node.y)
            
            # 如果到达终点，回溯路径并返回
            if current_pos == goal:
                print(f"找到路径！共搜索了 {iteration} 个节点")
                path = self.reconstruct_path(current_node)
                return path, visited_nodes
            
            # 将当前节点加入closed列表（标记为已访问）
            closed_list.add(current_pos)
            visited_nodes.append(current_pos)
            
            # 扩展邻居节点
            neighbors = self.get_neighbors(current_node.x, current_node.y)
            for neighbor_pos in neighbors:
                # 如果邻居已在closed列表中，跳过
                if neighbor_pos in closed_list:
                    continue
                
                # 计算从当前节点到邻居的代价
                move_cost = self.calculate_cost(current_node.x, current_node.y, 
                                              neighbor_pos[0], neighbor_pos[1])
                new_g = current_node.g + move_cost
                
                # 计算启发式估计
                new_h = self.heuristic(neighbor_pos[0], neighbor_pos[1], goal[0], goal[1])
                
                # 如果邻居不在node_info中，或发现更优路径
                if neighbor_pos not in node_info or new_g < node_info[neighbor_pos].g:
                    # 创建新节点
                    new_node = Node(
                        x=neighbor_pos[0],
                        y=neighbor_pos[1],
                        g=new_g,
                        h=new_h,
                        parent=current_node
                    )
                    
                    # 更新节点信息
                    node_info[neighbor_pos] = new_node
                    
                    # 加入open列表
                    heapq.heappush(open_list, new_node)
        
        # 如果open列表为空仍未找到路径
        print(f"无法找到路径！共搜索了 {iteration} 个节点")
        return None, visited_nodes
    
    def reconstruct_path(self, end_node):
        """
        回溯路径
        
        从终点节点开始，沿着父节点指针回溯到起点，
        然后反转路径得到从起点到终点的顺序。
        
        参数：
            end_node: 终点节点
        
        返回：
            path: 路径点列表
        """
        path = []
        current_node = end_node
        
        # 从终点回溯到起点
        while current_node is not None:
            path.append((current_node.x, current_node.y))
            current_node = current_node.parent
        
        # 反转路径，使其从起点到终点
        path.reverse()
        print(f"路径长度: {len(path)} 个节点")
        
        return path

# ==============================================================================
# 第四步：主程序 - 演示如何使用A*算法
# ==============================================================================

def main():
    """
    主函数：演示A*路径规划的完整流程
    
    教学步骤：
    1. 创建地图并添加障碍物
    2. 设置起点和终点
    3. 运行A*算法
    4. 可视化结果
    """
    
    # --------------------------
    # 步骤1: 创建地图
    # --------------------------
    print("=" * 50)
    print("步骤1: 创建20x15的网格地图")
    map = GridMap(width=20, height=15)
    
    # --------------------------
    # 步骤2: 添加障碍物
    # --------------------------
    print("\n步骤2: 添加障碍物")
    
    # 添加一些矩形障碍物（模拟建筑物或禁区）
    # 障碍物1: 中间偏左的矩形
    map.add_obstacle_rect(5, 3, 7, 8)
    print("  - 添加障碍物1: 矩形区域 (5,3)-(7,8)")
    
    # 障碍物2: 中间偏右的矩形
    map.add_obstacle_rect(12, 5, 14, 10)
    print("  - 添加障碍物2: 矩形区域 (12,5)-(14,10)")
    
    # 添加一些零散障碍物
    map.add_obstacle(10, 7)
    map.add_obstacle(10, 8)
    print("  - 添加障碍物3: 两个单独点 (10,7) 和 (10,8)")
    
    # --------------------------
    # 步骤3: 设置起点和终点
    # --------------------------
    print("\n步骤3: 设置起点和终点")
    start = (0, 0)    # 起点：地图左下角
    goal = (19, 14)   # 终点：地图右上角
    print(f"  - 起点: {start}")
    print(f"  - 终点: {goal}")
    
    # --------------------------
    # 步骤4: 运行A*算法
    # --------------------------
    print("\n步骤4: 运行A*路径规划算法")
    planner = AStarPlanner(map)
    path, visited_nodes = planner.plan(start, goal)
    
    # --------------------------
    # 步骤5: 可视化结果
    # --------------------------
    print("\n步骤5: 可视化路径规划结果")
    if path:
        map.plot(path, start, goal, visited_nodes)
        print("  [OK] Path planning succeeded!")
        print(f"  Path waypoints: {path}")
    else:
        print("  [FAIL] Cannot find path")
    
    print("\n" + "=" * 50)
    print("A*路径规划演示完成！")
    print("接下来可以学习：")
    print("  1. PSO路径优化")
    print("  2. B-spline轨迹平滑")
    print("  3. Gazebo无人机仿真")

if __name__ == '__main__':
    main()