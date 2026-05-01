# 无人机路径规划入门教程

> 浙江大学高飞无人机团队教学材料

本项目包含完整的无人机路径规划算法链，从全局路径搜索到轨迹生成，再到仿真验证。

## 📁 项目结构

```
无人机路径规划入门/
├── 01_AStar路径规划.py      # A*全局路径搜索
├── 02_PSO路径优化.py        # PSO路径点优化
├── 03_BSpline轨迹平滑.py    # B-spline连续轨迹生成
├── 04_Gazebo无人机仿真.md   # Gazebo仿真教程
└── README.md                # 项目说明
```

## 🚀 学习路径

| 序号 | 模块 | 学习目标 | 难度 |
|------|------|----------|------|
| 1 | A*路径规划 | 理解启发式搜索原理 | ⭐⭐ |
| 2 | PSO路径优化 | 掌握粒子群优化算法 | ⭐⭐⭐ |
| 3 | B-spline平滑 | 学习轨迹生成技术 | ⭐⭐⭐ |
| 4 | Gazebo仿真 | 实践无人机控制 | ⭐⭐⭐⭐ |

---

## 第1章：A*路径规划

### 1.1 算法原理

A*是一种启发式搜索算法，核心公式：

$$f(n) = g(n) + h(n)$$

- **g(n)**：从起点到节点n的实际代价
- **h(n)**：从节点n到终点的估计代价（启发函数）
- **f(n)**：总代价

### 1.2 运行代码

```bash
python 01_AStar路径规划.py
```

### 1.3 输出结果

- `AStar路径规划.png`：路径可视化图
- 路径点序列（用于后续优化）

### 1.4 核心代码解读

```python
# 启发函数（欧几里得距离）
def heuristic(self, x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

# 主循环：选择f值最小的节点扩展
while open_list:
    current_node = heapq.heappop(open_list)  # 优先队列
    # 扩展邻居...
```

---

## 第2章：PSO路径优化

### 2.1 算法原理

粒子群优化通过模拟鸟群觅食行为寻找最优解：

**速度更新公式**：
$$v_i = w \cdot v_i + c_1 \cdot r_1 \cdot (pbest_i - x_i) + c_2 \cdot r_2 \cdot (gbest - x_i)$$

- **w**：惯性权重
- **c1, c2**：认知和社会系数
- **r1, r2**：随机因子

### 2.2 运行代码

```bash
python 02_PSO路径优化.py
```

### 2.3 输出结果

- `PSO路径优化对比.png`：原始路径与优化路径对比
- 优化后的路径点

### 2.4 核心代码解读

```python
# 更新粒子速度
def update_velocity(self, gbest_position, w=0.5, c1=1.5, c2=1.5):
    r1 = np.random.random(self.position.shape)
    r2 = np.random.random(self.position.shape)
    cognitive = c1 * r1 * (self.pbest_position - self.position)
    social = c2 * r2 * (gbest_position - self.position)
    self.velocity = w * self.velocity + cognitive + social
```

---

## 第3章：B-spline轨迹平滑

### 3.1 算法原理

B-spline曲线将离散点拟合成光滑曲线，具有：
- **局部性**：修改一个控制点只影响局部
- **光滑性**：连续的一阶、二阶导数
- **灵活性**：可调整控制点形状

### 3.2 运行代码

```bash
python 03_BSpline轨迹平滑.py
```

### 3.3 输出结果

- `BSpline轨迹平滑.png`：轨迹对比图
- `BSpline速度分布.png`：速度分布图

### 3.4 核心代码解读

```python
# 创建B-spline插值器
spline = make_interp_spline(t, points, k=3)  # 3阶B-spline

# 生成平滑轨迹
t_eval = np.linspace(0, 1, num_points)
trajectory = spline(t_eval)
```

---

## 第4章：Gazebo无人机仿真

### 4.1 环境搭建

```bash
# 安装依赖
sudo apt install ros-noetic-gazebo-ros-pkgs ros-noetic-mavros

# 克隆rotors_simulator
git clone https://github.com/ethz-asl/rotors_simulator.git
```

### 4.2 启动仿真

```bash
# 启动无人机
roslaunch rotors_gazebo mav_hovering_example.launch

# 运行轨迹跟踪控制器
rosrun drone_control trajectory_tracker.py
```

### 4.3 控制器架构

```
期望轨迹 → PID控制 → 期望位姿 → 无人机执行
     ↑                  |
     |                  ↓
  状态反馈 ←——— 当前位姿
```

---

## 📊 算法性能对比

| 指标 | A*路径 | PSO优化后 | B-spline平滑后 |
|------|--------|-----------|----------------|
| 路径长度 | 32.5 | 28.3 (-12.9%) | 28.5 (+0.7%) |
| 拐点数量 | 5 | 5 | 0（连续） |
| 速度连续性 | 不连续 | 不连续 | 连续 |
| 加速度连续性 | 不连续 | 不连续 | 连续 |

---

## 🧪 实验建议

### 基础实验

1. 修改A*的启发函数（曼哈顿距离 vs 欧几里得距离）
2. 调整PSO的粒子数量和迭代次数
3. 尝试不同阶数的B-spline（2阶、3阶、4阶）

### 进阶实验

1. 添加障碍物动态避障功能
2. 实现时间最优轨迹规划
3. 在Gazebo中测试真实轨迹跟踪

---

## 📚 参考资料

1. *Probabilistic Robotics* - Sebastian Thrun
2. *Multiple View Geometry in Computer Vision* - Hartley & Zisserman
3. rotors_simulator官方文档：https://github.com/ethz-asl/rotors_simulator

---

## 📧 联系方式

浙江大学高飞无人机团队

如有问题或建议，请联系我们！