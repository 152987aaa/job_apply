#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浙大高飞无人机团队 | ROS基础学习模块
教学目标：掌握ROS话题、服务、坐标变换
对应论文：《ROS: An Open-Source Robot Operating System》
"""

import rospy
import numpy as np
from std_msgs.msg import String, Float64
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from custom_msgs.srv import GetDistance, GetDistanceResponse

class ROSBasicDemo:
    def __init__(self):
        """初始化ROS节点和组件"""
        rospy.init_node('ros_basic_demo', anonymous=True)
        
        # 1. 话题发布器
        self.cmd_pub = rospy.Publisher('/drone/cmd_vel', Float64, queue_size=10)
        self.pose_pub = rospy.Publisher('/drone/pose', PoseStamped, queue_size=10)
        
        # 2. 话题订阅器
        rospy.Subscriber('/drone/odometry', Odometry, self.odom_callback)
        rospy.Subscriber('/drone/status', String, self.status_callback)
        
        # 3. 服务服务器
        self.distance_service = rospy.Service('/drone/get_distance', GetDistance, self.get_distance_callback)
        
        # 4. 坐标变换广播器
        self.tf_broadcaster = TransformBroadcaster()
        
        # 5. 坐标变换监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        
        # 状态变量
        self.current_pose = None
        self.rate = rospy.Rate(10)  # 10Hz
        
        rospy.loginfo("ROS基础演示节点已启动")
    
    def odom_callback(self, msg):
        """里程计回调函数"""
        self.current_pose = msg.pose.pose
        rospy.logdebug(f"收到里程计数据: x={msg.pose.pose.position.x:.2f}")
    
    def status_callback(self, msg):
        """状态回调函数"""
        rospy.loginfo(f"无人机状态: {msg.data}")
    
    def get_distance_callback(self, req):
        """距离计算服务回调"""
        if self.current_pose is None:
            return GetDistanceResponse(-1.0)
        
        dx = req.target_x - self.current_pose.position.x
        dy = req.target_y - self.current_pose.position.y
        distance = np.sqrt(dx**2 + dy**2)
        
        return GetDistanceResponse(distance)
    
    def broadcast_transform(self):
        """广播坐标变换"""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = "drone/base_link"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 2.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)
    
    def run(self):
        """主循环"""
        while not rospy.is_shutdown():
            # 发布速度指令
            cmd_msg = Float64()
            cmd_msg.data = 1.0
            self.cmd_pub.publish(cmd_msg)
            
            # 发布位姿信息
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position.x = 1.0
            pose_msg.pose.position.y = 2.0
            pose_msg.pose.position.z = 3.0
            self.pose_pub.publish(pose_msg)
            
            # 广播坐标变换
            self.broadcast_transform()
            
            # 尝试获取坐标变换
            try:
                trans = self.tf_buffer.lookup_transform(
                    "world", "drone/base_link", rospy.Time(0)
                )
                rospy.logdebug(f"TF变换: x={trans.transform.translation.x}")
            except Exception as e:
                rospy.logdebug(f"TF获取失败: {e}")
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        demo = ROSBasicDemo()
        demo.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS基础演示节点已停止")