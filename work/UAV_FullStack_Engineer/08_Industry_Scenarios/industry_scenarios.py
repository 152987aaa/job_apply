#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浙大高飞无人机团队 | 行业场景模块
教学目标：掌握电力巡检、物流配送、矿洞巡检场景
对应标准：《无人机电力巡检技术规范》
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PowerInspection:
    """电力巡检场景"""
    
    def __init__(self):
        """初始化电力巡检场景"""
        self.towers = []  # 杆塔位置
        self.lines = []   # 线路连接
        self.inspection_points = []  # 巡检点
    
    def load_power_grid(self, file_path=None):
        """加载电网数据"""
        # 模拟电网数据
        print("加载电网数据...")
        
        # 创建模拟杆塔
        self.towers = [
            (0, 0), (5, 2), (10, 0), (15, 3), (20, 0),
            (25, 4), (30, 1), (35, 5), (40, 2), (45, 0)
        ]
        
        # 创建线路连接
        self.lines = [(i, i+1) for i in range(len(self.towers)-1)]
        
        # 生成巡检点（每个杆塔周围的检查点）
        for i, tower in enumerate(self.towers):
            for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                offset_x = 1.5 * np.cos(angle)
                offset_y = 1.5 * np.sin(angle)
                self.inspection_points.append({
                    'x': tower[0] + offset_x,
                    'y': tower[1] + offset_y,
                    'tower_id': i,
                    'type': 'inspection'
                })
        
        print(f"电网数据加载完成，共 {len(self.towers)} 个杆塔，{len(self.inspection_points)} 个巡检点")
    
    def plan_inspection_route(self, start_point=(0, 0)):
        """规划巡检路线（TSP问题简化）"""
        print("规划巡检路线...")
        
        # 简化的最近邻算法
        remaining_points = self.inspection_points.copy()
        route = [start_point]
        current_point = start_point
        
        while remaining_points:
            # 找到最近的未访问点
            min_dist = float('inf')
            next_point = None
            
            for point in remaining_points:
                dist = np.hypot(point['x'] - current_point[0], point['y'] - current_point[1])
                if dist < min_dist:
                    min_dist = dist
                    next_point = point
            
            if next_point:
                route.append((next_point['x'], next_point['y']))
                remaining_points.remove(next_point)
                current_point = (next_point['x'], next_point['y'])
        
        # 返回起点
        route.append(start_point)
        
        print(f"巡检路线规划完成，共 {len(route)} 个航点")
        return route
    
    def visualize_grid(self, route=None):
        """可视化电网和巡检路线"""
        plt.figure(figsize=(12, 6))
        
        # 绘制线路
        for line in self.lines:
            t1 = self.towers[line[0]]
            t2 = self.towers[line[1]]
            plt.plot([t1[0], t2[0]], [t1[1], t2[1]], 'k-', linewidth=2, label='输电线路')
        
        # 绘制杆塔
        tower_x = [t[0] for t in self.towers]
        tower_y = [t[1] for t in self.towers]
        plt.scatter(tower_x, tower_y, c='red', s=100, marker='^', label='杆塔')
        
        # 绘制巡检点
        inspect_x = [p['x'] for p in self.inspection_points]
        inspect_y = [p['y'] for p in self.inspection_points]
        plt.scatter(inspect_x, inspect_y, c='blue', s=50, label='巡检点')
        
        # 绘制巡检路线
        if route:
            route_x = [p[0] for p in route]
            route_y = [p[1] for p in route]
            plt.plot(route_x, route_y, 'g--', linewidth=2, label='巡检路线')
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('电力巡检场景')
        plt.legend()
        plt.grid(True)
        plt.savefig('power_inspection.png')
        plt.close()
        print("电力巡检可视化图已保存")

class LogisticsDelivery:
    """物流配送场景"""
    
    def __init__(self):
        """初始化物流配送场景"""
        self.depot = None       # 仓库位置
        self.customers = []     # 客户位置
        self.obstacles = []     # 障碍物
    
    def setup_scenario(self):
        """设置配送场景"""
        print("设置物流配送场景...")
        
        # 设置仓库位置
        self.depot = (10, 10)
        
        # 设置客户位置
        self.customers = [
            (5, 5), (15, 5), (20, 10), (15, 15), (5, 15),
            (8, 8), (12, 8), (12, 12), (8, 12), (18, 12)
        ]
        
        # 设置障碍物（建筑物）
        self.obstacles = [
            {'x': 10, 'y': 0, 'width': 4, 'height': 3},
            {'x': 0, 'y': 10, 'width': 3, 'height': 4},
            {'x': 17, 'y': 17, 'width': 5, 'height': 3}
        ]
        
        print(f"场景设置完成，仓库: {self.depot}，客户数: {len(self.customers)}")
    
    def plan_delivery_route(self, max_stops=5):
        """规划配送路线"""
        print("规划配送路线...")
        
        routes = []
        remaining_customers = self.customers.copy()
        
        while remaining_customers:
            route = [self.depot]
            current_point = self.depot
            
            for _ in range(max_stops):
                if not remaining_customers:
                    break
                
                # 最近邻选择
                min_dist = float('inf')
                next_customer = None
                
                for customer in remaining_customers:
                    dist = np.hypot(customer[0] - current_point[0], customer[1] - current_point[1])
                    if dist < min_dist:
                        min_dist = dist
                        next_customer = customer
                
                if next_customer:
                    route.append(next_customer)
                    remaining_customers.remove(next_customer)
                    current_point = next_customer
            
            route.append(self.depot)
            routes.append(route)
        
        print(f"配送路线规划完成，共 {len(routes)} 条路线")
        return routes
    
    def visualize_delivery(self, routes=None):
        """可视化配送场景"""
        plt.figure(figsize=(10, 10))
        
        # 绘制障碍物
        for obs in self.obstacles:
            rect = plt.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'], 
                                color='gray', alpha=0.5)
            plt.gca().add_patch(rect)
        
        # 绘制仓库
        plt.scatter(self.depot[0], self.depot[1], c='green', s=150, marker='s', label='仓库')
        
        # 绘制客户
        customer_x = [c[0] for c in self.customers]
        customer_y = [c[1] for c in self.customers]
        plt.scatter(customer_x, customer_y, c='red', s=80, label='客户')
        
        # 绘制配送路线
        if routes:
            colors = ['blue', 'orange', 'purple', 'brown']
            for i, route in enumerate(routes):
                route_x = [p[0] for p in route]
                route_y = [p[1] for p in route]
                plt.plot(route_x, route_y, '--', linewidth=2, color=colors[i % len(colors)],
                        label=f'路线{i+1}')
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('物流配送场景')
        plt.legend()
        plt.grid(True)
        plt.savefig('logistics_delivery.png')
        plt.close()
        print("物流配送可视化图已保存")

class MineInspection:
    """矿洞巡检场景"""
    
    def __init__(self):
        """初始化矿洞巡检场景"""
        self.tunnels = []      # 隧道网络
        self.checkpoints = []  # 检查点
        self.degenerate_zones = []  # 退化区域
    
    def load_mine_map(self):
        """加载矿洞地图"""
        print("加载矿洞地图...")
        
        # 创建隧道网络
        self.tunnels = [
            ((0, 0), (5, 0)),
            ((5, 0), (10, 0)),
            ((10, 0), (10, 5)),
            ((10, 5), (15, 5)),
            ((15, 5), (15, 10)),
            ((15, 10), (20, 10)),
            ((20, 10), (20, 5)),
            ((20, 5), (25, 5)),
            ((25, 5), (25, 0)),
            ((10, 0), (10, -5)),
            ((15, 5), (15, 0))
        ]
        
        # 设置检查点
        self.checkpoints = [
            {'x': 5, 'y': 0, 'type': 'gas_sensor'},
            {'x': 10, 'y': 5, 'type': 'camera'},
            {'x': 15, 'y': 10, 'type': 'gas_sensor'},
            {'x': 20, 'y': 5, 'type': 'camera'},
            {'x': 25, 'y': 0, 'type': 'emergency_exit'}
        ]
        
        # 设置退化区域（无GPS区域）
        self.degenerate_zones = [
            {'x_min': 12, 'x_max': 18, 'y_min': 8, 'y_max': 12},
            {'x_min': 22, 'x_max': 26, 'y_min': -2, 'y_max': 2}
        ]
        
        print(f"矿洞地图加载完成，隧道数: {len(self.tunnels)}, 检查点: {len(self.checkpoints)}")
    
    def plan_mine_route(self, start=(0, 0)):
        """规划矿洞巡检路线"""
        print("规划矿洞巡检路线...")
        
        # 简单路线规划：遍历所有检查点
        route = [start]
        
        # 按顺序访问检查点
        for checkpoint in self.checkpoints:
            route.append((checkpoint['x'], checkpoint['y']))
        
        # 返回起点
        route.append(start)
        
        print("矿洞巡检路线规划完成")
        return route
    
    def visualize_mine(self, route=None):
        """可视化矿洞场景"""
        plt.figure(figsize=(12, 8))
        
        # 绘制隧道
        for tunnel in self.tunnels:
            plt.plot([tunnel[0][0], tunnel[1][0]], 
                     [tunnel[0][1], tunnel[1][1]], 
                     'k-', linewidth=4, label='隧道' if tunnel == self.tunnels[0] else "")
        
        # 绘制退化区域
        for zone in self.degenerate_zones:
            rect = plt.Rectangle((zone['x_min'], zone['y_min']), 
                                zone['x_max'] - zone['x_min'], 
                                zone['y_max'] - zone['y_min'],
                                color='yellow', alpha=0.3)
            plt.gca().add_patch(rect)
        
        # 绘制检查点
        for checkpoint in self.checkpoints:
            plt.scatter(checkpoint['x'], checkpoint['y'], 
                       c='red' if checkpoint['type'] == 'gas_sensor' else 'blue', 
                       s=100, label=checkpoint['type'])
        
        # 绘制路线
        if route:
            route_x = [p[0] for p in route]
            route_y = [p[1] for p in route]
            plt.plot(route_x, route_y, 'g--', linewidth=2, label='巡检路线')
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('矿洞巡检场景')
        plt.legend()
        plt.grid(True)
        plt.savefig('mine_inspection.png')
        plt.close()
        print("矿洞巡检可视化图已保存")

def test_scenarios():
    """测试各行业场景"""
    print("测试电力巡检场景...")
    power = PowerInspection()
    power.load_power_grid()
    route = power.plan_inspection_route()
    power.visualize_grid(route)
    
    print("\n测试物流配送场景...")
    logistics = LogisticsDelivery()
    logistics.setup_scenario()
    routes = logistics.plan_delivery_route()
    logistics.visualize_delivery(routes)
    
    print("\n测试矿洞巡检场景...")
    mine = MineInspection()
    mine.load_mine_map()
    route = mine.plan_mine_route()
    mine.visualize_mine(route)
    
    print("\n行业场景测试完成！")

if __name__ == '__main__':
    test_scenarios()