# 无人机路径规划教学文档

> 浙大高飞无人机团队 | 理论+代码+仿真全闭环教学

---

## 一、学习路线图

```
第1周：A*全局路径规划
    ↓
第2周：PSO路径优化
    ↓
第3周：B-spline轨迹平滑
    ↓
第4周：Gazebo仿真验证
```

---

## 二、A*路径规划

### 2.1 核心原理

A*是一种启发式搜索算法，核心公式：

$$f(n) = g(n) + h(n)$$

- **g(n)**：从起点到节点n的实际代价
- **h(n)**：从节点n到终点的估计代价（启发函数）
- **f(n)**：总代价

### 2.2 算法流程

```
1. 将起点加入open列表
2. 选择f值最小的节点扩展
3. 将节点移到closed列表
4. 扩展8个方向的邻居
5. 如果邻居不在closed且更优，加入open列表
6. 重复直到找到终点或open为空
```

### 2.3 关键代码解析

```python
# 启发函数：欧氏距离
def heuristic(self, node):
    return np.hypot(node[0]-self.goal[0], node[1]-self.goal[1])

# 主循环：优先队列扩展
while open_heap:
    f_cost, g_cost, current_node, parent = heapq.heappop(open_heap)
    if current_node == goal:
        # 回溯路径
        path = []
        while current_node:
            path.append(current_node)
            current_node = parent_dict.get(current_node)
        return path[::-1]
```

### 2.4 练习题目

1. 修改启发函数为曼哈顿距离，观察路径变化
2. 添加动态障碍物，测试A*的重规划能力
3. 调整地图尺寸和障碍物位置，分析规划时间

---

## 三、PSO路径优化

### 3.1 核心原理

粒子群优化模拟鸟群觅食行为：

**速度更新公式**：
$$v_i = w \cdot v_i + c_1 \cdot r_1 \cdot (pbest_i - x_i) + c_2 \cdot r_2 \cdot (gbest - x_i)$$

### 3.2 参数说明

| 参数 | 符号 | 作用 | 推荐值 |
|------|------|------|--------|
| 惯性权重 | w | 控制历史速度影响 | 0.5 |
| 认知系数 | c1 | 个人经验权重 | 1.5 |
| 社会系数 | c2 | 群体经验权重 | 1.5 |

### 3.3 关键代码解析

```python
def update_velocity(self, gbest_position):
    r1 = np.random.random(self.position.shape)
    r2 = np.random.random(self.position.shape)
    cognitive = c1 * r1 * (self.pbest - self.position)
    social = c2 * r2 * (gbest_position - self.position)
    self.velocity = w * self.velocity + cognitive + social
```

### 3.4 练习题目

1. 调整粒子数量和迭代次数，观察收敛速度
2. 修改惯性权重为递减策略，分析效果
3. 添加路径平滑约束（如最大曲率限制）

---

## 四、B-spline轨迹平滑

### 4.1 核心原理

B-spline曲线将离散点拟合成光滑曲线：

$$C(t) = \sum_{i=0}^n P_i \cdot N_{i,k}(t)$$

其中 $N_{i,k}(t)$ 是k阶B样条基函数。

### 4.2 参数化方法

使用累积距离参数化：

```python
def _compute_parameterization(self):
    distances = np.zeros(len(self.waypoints))
    for i in range(1, len(self.waypoints)):
        distances[i] = np.hypot(x[i]-x[i-1], y[i]-y[i-1])
    t = np.cumsum(distances) / distances[-1]
    return t
```

### 4.3 练习题目

1. 尝试不同阶数的B-spline（2阶、3阶、4阶）
2. 添加时间约束，生成时间最优轨迹
3. 分析速度和加速度的连续性

---

## 五、Gazebo仿真

### 5.1 环境搭建

```bash
# 安装依赖
sudo apt install ros-noetic-gazebo-ros ros-noetic-rotors-simulator

# 启动仿真
roslaunch rotors_gazebo mav_hovering_example.launch
```

### 5.2 控制器架构

```
期望轨迹 → PID控制 → 期望位姿 → 无人机执行
     ↑                  |
     |                  ↓
  状态反馈 ←——— 当前位姿
```

### 5.3 ROS话题列表

| 话题 | 类型 | 说明 |
|------|------|------|
| `/firefly/ground_truth/pose` | PoseStamped | 真实位姿 |
| `/firefly/command/pose` | PoseStamped | 期望位姿 |
| `/firefly/imu` | Imu | IMU数据 |

### 5.4 练习题目

1. 调整PID参数，优化跟踪精度
2. 添加避障功能（使用激光雷达）
3. 实现自主起飞→跟踪→降落流程

---

## 六、常见问题

### 6.1 A*找不到路径
- 检查起点/终点是否在障碍物中
- 检查障碍物是否完全阻挡了路径
- 尝试增大地图尺寸

### 6.2 PSO不收敛
- 增加粒子数量或迭代次数
- 调整惯性权重和学习系数
- 检查边界设置是否合理

### 6.3 B-spline过拟合
- 减少路点数量
- 使用更高阶的B-spline
- 添加正则化约束

### 6.4 Gazebo启动失败
- 检查ROS环境变量是否正确
- 检查显卡驱动是否安装
- 尝试重新编译工作空间

---

## 七、进阶扩展

### 7.1 三维路径规划
- 在A*中添加Z轴维度
- 使用八叉树或体素地图
- 考虑无人机飞行高度约束

### 7.2 动态避障
- 结合激光雷达数据
- 使用局部避障算法（如APF）
- 实现实时重规划

### 7.3 多机协同
- 使用分布式路径规划
- 添加碰撞避免约束
- 实现编队飞行

---

## 八、参考文献

1. *Probabilistic Robotics* - Sebastian Thrun
2. *Multiple View Geometry* - Hartley & Zisserman
3. rotors_simulator官方文档：https://github.com/ethz-asl/rotors_simulator