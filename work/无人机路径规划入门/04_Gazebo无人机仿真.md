# Gazebo无人机仿真与轨迹跟踪 - 浙江大学高飞无人机团队教学材料

## 一、实验环境搭建

### 1.1 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Ubuntu 20.04 / 18.04 |
| ROS版本 | Noetic / Melodic |
| 硬件 | 推荐GPU（NVIDIA显卡） |
| 内存 | 至少8GB |

### 1.2 安装步骤

```bash
# 1. 安装ROS（以Noetic为例）
sudo apt update && sudo apt install ros-noetic-desktop-full

# 2. 安装rotors_simulator依赖
sudo apt install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
sudo apt install ros-noetic-joy ros-noetic-octomap-ros ros-noetic-mavros

# 3. 创建工作空间
mkdir -p ~/drone_ws/src
cd ~/drone_ws/src

# 4. 克隆rotors_simulator
git clone https://github.com/ethz-asl/rotors_simulator.git

# 5. 克隆相关依赖
git clone https://github.com/ethz-asl/mav_comm.git
git clone https://github.com/ethz-asl/glog_catkin.git
git clone https://github.com/catkin/catkin_simple.git

# 6. 编译
cd ~/drone_ws
catkin_make -DCMAKE_BUILD_TYPE=Release

# 7. 配置环境
echo "source ~/drone_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 1.3 测试安装

```bash
# 启动仿真环境
roslaunch rotors_gazebo mav_hovering_example.launch mav_name:=firefly
```

如果一切正常，你会看到Gazebo启动并显示一架无人机。

---

## 二、rotors_simulator架构解析

### 2.1 核心组件

```
┌─────────────────────────────────────────────────────────────────┐
│                      Gazebo仿真环境                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │  无人机   │    │  传感器   │    │   环境    │               │
│  │  模型     │    │  (IMU等)  │    │  (地图等) │               │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘               │
│       │               │               │                        │
├───────┼───────────────┼───────────────┼──────────────────────┤
│       ▼               ▼               ▼                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                   ROS消息系统                         │    │
│  │  /firefly/command/motor_speed   (电机控制指令)       │    │
│  │  /firefly/imu                   (IMU数据)           │    │
│  │  /firefly/ground_truth/pose     (真实位姿)           │    │
│  └────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐    │
│  │                   控制器节点                          │    │
│  │  attitude_controller (姿态控制)                      │    │
│  │  position_controller (位置控制)                      │    │
│  │  trajectory_controller (轨迹跟踪控制器)              │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键话题

| 话题名称 | 消息类型 | 说明 |
|----------|----------|------|
| `/firefly/command/motor_speed` | `mav_msgs/Actuators` | 电机转速指令 |
| `/firefly/imu` | `sensor_msgs/Imu` | IMU传感器数据 |
| `/firefly/ground_truth/pose` | `geometry_msgs/PoseStamped` | 无人机真实位姿 |
| `/firefly/ground_truth/twist` | `geometry_msgs/TwistStamped` | 无人机真实速度 |
| `/firefly/command/pose` | `geometry_msgs/PoseStamped` | 期望位姿指令 |
| `/firefly/command/trajectory` | `mav_msgs/CommandTrajectory` | 期望轨迹指令 |

---

## 三、轨迹跟踪控制器实现

### 3.1 控制器架构

```
┌────────────────────────────────────────────────────────┐
│              轨迹跟踪控制器                           │
├────────────────────────────────────────────────────────┤
│  输入:                                                │
│    - 期望轨迹 (位置、速度、加速度)                      │
│    - 当前状态 (位置、速度)                             │
│                                                       │
│  输出:                                                │
│    - 期望姿态角 (roll, pitch, yaw)                     │
│    - 期望高度 (z)                                     │
│                                                       │
│  控制律:                                              │
│    位置误差 → PID控制器 → 速度指令                    │
│    速度误差 → PID控制器 → 姿态角指令                  │
└────────────────────────────────────────────────────────┘
```

### 3.2 Python控制器代码

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机轨迹跟踪控制器 - 浙江大学高飞无人机团队

使用PID控制实现轨迹跟踪
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from mav_msgs.msg import CommandTrajectory

class TrajectoryTracker:
    def __init__(self):
        # ROS节点初始化
        rospy.init_node('trajectory_tracker', anonymous=True)
        
        # 订阅当前状态
        rospy.Subscriber('/firefly/ground_truth/pose', 
                         PoseStamped, self.pose_callback)
        rospy.Subscriber('/firefly/ground_truth/twist', 
                         TwistStamped, self.twist_callback)
        
        # 发布控制指令
        self.pose_pub = rospy.Publisher('/firefly/command/pose', 
                                        PoseStamped, queue_size=10)
        
        # 当前状态
        self.current_pose = None
        self.current_velocity = None
        
        # PID参数
        self.kp_pos = np.array([2.0, 2.0, 2.0])  # 位置比例系数
        self.ki_pos = np.array([0.0, 0.0, 0.0])  # 位置积分系数
        self.kd_pos = np.array([0.5, 0.5, 0.5])  # 位置微分系数
        
        # 积分误差累积
        self.pos_integral = np.zeros(3)
        
        # 期望轨迹（预定义的B-spline轨迹点）
        self.trajectory_points = self.load_trajectory()
        self.trajectory_index = 0
        
        # 控制频率
        self.rate = rospy.Rate(100)  # 100Hz
    
    def pose_callback(self, msg):
        """接收当前位姿"""
        self.current_pose = msg.pose
    
    def twist_callback(self, msg):
        """接收当前速度"""
        self.current_velocity = msg.twist
    
    def load_trajectory(self):
        """加载预计算的B-spline轨迹"""
        # 这里应该从文件或参数服务器加载轨迹
        # 以下是模拟数据
        trajectory = []
        t = np.linspace(0, 10, 100)
        for i in range(100):
            x = 2.0 * t[i]
            y = 1.0 * np.sin(t[i])
            z = 2.0 + 0.5 * np.cos(t[i])
            trajectory.append((x, y, z))
        return trajectory
    
    def pid_control(self, desired_pos, current_pos, current_vel, dt):
        """
        PID控制器
        
        参数:
            desired_pos: 期望位置 [x, y, z]
            current_pos: 当前位置 [x, y, z]
            current_vel: 当前速度 [vx, vy, vz]
            dt: 时间间隔
        
        返回:
            desired_vel: 期望速度
        """
        # 计算位置误差
        pos_error = np.array(desired_pos) - np.array(current_pos)
        
        # 累积积分误差
        self.pos_integral += pos_error * dt
        
        # 计算速度误差（假设期望速度为0，只做位置控制）
        vel_error = -np.array(current_vel)
        
        # PID控制律
        desired_vel = (self.kp_pos * pos_error + 
                       self.ki_pos * self.pos_integral + 
                       self.kd_pos * vel_error)
        
        return desired_vel
    
    def run(self):
        """主控制循环"""
        rospy.loginfo("轨迹跟踪控制器启动")
        
        while not rospy.is_shutdown():
            if self.current_pose is None or self.current_velocity is None:
                self.rate.sleep()
                continue
            
            # 获取当前位置
            current_pos = [
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z
            ]
            
            # 获取当前速度
            current_vel = [
                self.current_velocity.linear.x,
                self.current_velocity.linear.y,
                self.current_velocity.linear.z
            ]
            
            # 获取期望位置（从轨迹中采样）
            if self.trajectory_index < len(self.trajectory_points):
                desired_pos = self.trajectory_points[self.trajectory_index]
                self.trajectory_index += 1
            else:
                # 轨迹结束，悬停在最后一个点
                desired_pos = self.trajectory_points[-1]
            
            # 计算控制输出
            dt = 0.01  # 100Hz控制频率
            desired_vel = self.pid_control(desired_pos, current_pos, current_vel, dt)
            
            # 发布期望位姿
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = 'world'
            
            # 设置位置
            pose_msg.pose.position.x = desired_pos[0]
            pose_msg.pose.position.y = desired_pos[1]
            pose_msg.pose.position.z = desired_pos[2]
            
            # 设置姿态（简单设置为水平）
            pose_msg.pose.orientation.w = 1.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            
            self.pose_pub.publish(pose_msg)
            
            # 打印调试信息
            if self.trajectory_index % 10 == 0:
                rospy.loginfo(f"跟踪进度: {self.trajectory_index}/{len(self.trajectory_points)}")
                rospy.loginfo(f"当前位置: {current_pos}")
                rospy.loginfo(f"期望位置: {desired_pos}")
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        tracker = TrajectoryTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("轨迹跟踪控制器停止")
```

### 3.3 启动脚本

创建 `trajectory_tracking.launch` 文件：

```xml
<?xml version="1.0"?>
<launch>
    <!-- 启动Gazebo仿真环境 -->
    <include file="$(find rotors_gazebo)/launch/mav_hovering_example.launch">
        <arg name="mav_name" value="firefly" />
    </include>
    
    <!-- 启动轨迹跟踪控制器 -->
    <node name="trajectory_tracker" 
          pkg="drone_control" 
          type="trajectory_tracker.py" 
          output="screen">
        <!-- PID参数 -->
        <param name="kp_x" value="2.0" />
        <param name="kp_y" value="2.0" />
        <param name="kp_z" value="2.0" />
    </node>
</launch>
```

---

## 四、完整实验流程

### 4.1 步骤1: 启动仿真环境

```bash
# 终端1: 启动Gazebo和无人机
roslaunch rotors_gazebo mav_hovering_example.launch mav_name:=firefly
```

### 4.2 步骤2: 启动轨迹跟踪控制器

```bash
# 终端2: 运行控制器
rosrun drone_control trajectory_tracker.py
```

### 4.3 步骤3: 观察无人机行为

在Gazebo中观察无人机：
1. 无人机应该起飞并悬停
2. 然后按照预定义的轨迹飞行
3. 终端会输出跟踪进度和位置信息

### 4.4 步骤4: 记录数据（可选）

```bash
# 终端3: 记录数据
rosbag record /firefly/ground_truth/pose /firefly/command/pose -O trajectory_data.bag
```

---

## 五、常见问题与调试

### 5.1 无人机飞不起来

**可能原因**:
- 电机速度指令没有正确发布
- 控制器参数设置不当
- 仿真环境没有正确加载

**解决方法**:
```bash
# 检查话题发布情况
rostopic list
rostopic echo /firefly/command/motor_speed
```

### 5.2 轨迹跟踪不稳定

**可能原因**:
- PID参数过大导致震荡
- 控制频率过低
- 噪声干扰

**解决方法**:
1. 减小比例系数Kp
2. 增加微分系数Kd
3. 提高控制频率

### 5.3 ROS节点通信问题

**检查通信链路**:
```bash
# 检查节点状态
rosnode list
rosnode info /trajectory_tracker

# 检查话题连接
rostopic info /firefly/command/pose
```

---

## 六、进阶练习

### 6.1 练习1: 优化PID参数

尝试调整PID参数，观察无人机的响应：

| 参数 | 作用 | 调整建议 |
|------|------|----------|
| Kp | 响应速度 | 增大加快响应，但过大可能震荡 |
| Ki | 消除稳态误差 | 用于处理重力等恒定干扰 |
| Kd | 抑制震荡 | 增大可减小震荡，但可能降低响应速度 |

### 6.2 练习2: 实现更复杂的轨迹

修改轨迹生成代码，实现螺旋上升轨迹：

```python
def load_trajectory():
    trajectory = []
    t = np.linspace(0, 20, 200)
    for i in range(200):
        r = 2.0 + t[i] * 0.1  # 螺旋半径随时间增加
        x = r * np.cos(t[i])
        y = r * np.sin(t[i])
        z = 1.0 + t[i] * 0.1  # 高度逐渐上升
        trajectory.append((x, y, z))
    return trajectory
```

### 6.3 练习3: 添加避障功能

结合之前的A*算法，在仿真中添加避障：
1. 使用Gazebo的激光雷达传感器
2. 实时检测障碍物
3. 动态调整轨迹

---

## 七、总结

通过本实验，你应该掌握：

1. **Gazebo仿真环境搭建**：如何安装和配置rotors_simulator
2. **ROS消息系统**：如何订阅和发布话题
3. **PID控制**：如何设计位置控制器
4. **轨迹跟踪**：如何让无人机跟随预定轨迹

接下来可以学习：
- 更高级的控制算法（如MPC模型预测控制）
- SLAM同时定位与地图构建
- 自主导航和避障

---

## 附录：常用命令速查

| 命令 | 说明 |
|------|------|
| `roslaunch rotors_gazebo mav_hovering_example.launch` | 启动无人机仿真 |
| `rostopic list` | 列出所有话题 |
| `rostopic echo /topic_name` | 查看话题内容 |
| `rosnode list` | 列出所有节点 |
| `rosbag record` | 记录数据 |
| `rviz` | 可视化工具 |