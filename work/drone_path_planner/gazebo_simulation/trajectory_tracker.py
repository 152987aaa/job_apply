"""
浙大高飞无人机团队 | Gazebo无人机轨迹跟踪控制器
【严格对应链接步骤4】
功能：读取B-spline轨迹 → 控制无人机跟踪 → ROS节点实现
教学重点：ROS消息系统、PID控制、无人机仿真
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from mav_msgs.msg import CommandTrajectory
from nav_msgs.msg import Path

class TrajectoryTracker:
    def __init__(self):
        """初始化轨迹跟踪控制器"""
        # ROS节点初始化
        rospy.init_node('trajectory_tracker', anonymous=True)
        
        # 订阅无人机状态
        rospy.Subscriber('/firefly/ground_truth/pose', 
                         PoseStamped, self.pose_callback)
        rospy.Subscriber('/firefly/ground_truth/twist', 
                         TwistStamped, self.twist_callback)
        
        # 发布控制指令
        self.pose_pub = rospy.Publisher('/firefly/command/pose', 
                                        PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher('/trajectory', 
                                        Path, queue_size=10)
        
        # 当前状态
        self.current_pose = None
        self.current_velocity = None
        
        # PID控制器参数
        self.kp_pos = np.array([2.0, 2.0, 2.0])   # 位置比例系数
        self.ki_pos = np.array([0.0, 0.0, 0.0])   # 位置积分系数
        self.kd_pos = np.array([0.5, 0.5, 0.5])   # 位置微分系数
        
        # 积分误差累积
        self.pos_integral = np.zeros(3)
        
        # 期望轨迹
        self.trajectory = None
        self.trajectory_index = 0
        
        # 控制频率
        self.rate = rospy.Rate(100)  # 100Hz
    
    def pose_callback(self, msg):
        """接收无人机当前位姿"""
        self.current_pose = msg.pose
    
    def twist_callback(self, msg):
        """接收无人机当前速度"""
        self.current_velocity = msg.twist
    
    def load_trajectory(self, file_path):
        """
        加载轨迹文件
        :param file_path: 轨迹文件路径
        """
        try:
            self.trajectory = np.loadtxt(file_path)
            print(f"成功加载轨迹文件: {file_path}")
            print(f"轨迹点数: {len(self.trajectory)}")
            return True
        except Exception as e:
            print(f"加载轨迹失败: {e}")
            return False
    
    def pid_control(self, desired_pos, current_pos, current_vel, dt):
        """
        PID控制器
        :param desired_pos: 期望位置 [x, y, z]
        :param current_pos: 当前位置 [x, y, z]
        :param current_vel: 当前速度 [vx, vy, vz]
        :param dt: 时间间隔
        :return: 控制输出
        """
        # 计算位置误差
        pos_error = np.array(desired_pos) - np.array(current_pos)
        
        # 累积积分误差
        self.pos_integral += pos_error * dt
        
        # 计算速度误差
        vel_error = -np.array(current_vel)
        
        # PID控制律
        control_output = (self.kp_pos * pos_error + 
                         self.ki_pos * self.pos_integral + 
                         self.kd_pos * vel_error)
        
        return control_output
    
    def run(self):
        """主控制循环"""
        rospy.loginfo("轨迹跟踪控制器启动")
        
        while not rospy.is_shutdown():
            # 检查状态是否就绪
            if self.current_pose is None or self.current_velocity is None:
                self.rate.sleep()
                continue
            
            if self.trajectory is None:
                rospy.logwarn("轨迹未加载")
                self.rate.sleep()
                continue
            
            # 获取当前状态
            current_pos = [
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z
            ]
            
            current_vel = [
                self.current_velocity.linear.x,
                self.current_velocity.linear.y,
                self.current_velocity.linear.z
            ]
            
            # 获取期望位置
            if self.trajectory_index < len(self.trajectory):
                desired_pos = self.trajectory[self.trajectory_index]
                # 添加Z轴高度
                desired_pos = [desired_pos[0], desired_pos[1], 2.0]
                self.trajectory_index += 1
            else:
                desired_pos = [self.trajectory[-1][0], self.trajectory[-1][1], 2.0]
            
            # 计算控制输出（这里简化为直接发布期望位置）
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = 'world'
            
            # 设置位置
            pose_msg.pose.position.x = desired_pos[0]
            pose_msg.pose.position.y = desired_pos[1]
            pose_msg.pose.position.z = desired_pos[2]
            
            # 设置姿态（水平姿态）
            pose_msg.pose.orientation.w = 1.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            
            self.pose_pub.publish(pose_msg)
            
            # 发布轨迹可视化
            self.publish_path()
            
            # 打印调试信息
            if self.trajectory_index % 50 == 0:
                rospy.loginfo(f"跟踪进度: {self.trajectory_index}/{len(self.trajectory)}")
            
            self.rate.sleep()
    
    def publish_path(self):
        """发布轨迹路径用于RViz可视化"""
        if self.trajectory is None:
            return
        
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = 'world'
        
        for point in self.trajectory:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = point[0]
            pose_stamped.pose.position.y = point[1]
            pose_stamped.pose.position.z = 2.0
            path_msg.poses.append(pose_stamped)
        
        self.path_pub.publish(path_msg)

if __name__ == '__main__':
    try:
        tracker = TrajectoryTracker()
        
        # 加载轨迹
        success = tracker.load_trajectory('../data/output/bspline_trajectory.txt')
        if not success:
            rospy.logerr("无法加载轨迹文件，退出")
            exit(1)
        
        # 启动跟踪
        tracker.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("轨迹跟踪控制器停止")