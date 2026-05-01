#!/bin/bash
# 浙大高飞无人机团队 | 环境安装脚本
# Ubuntu 20.04 + ROS Noetic + Python3 + Gazebo11 + rotors_simulator

echo "========== 浙大高飞无人机团队 - 环境安装脚本 =========="

# 检查是否为root用户
if [ "$(id -u)" != "0" ]; then
    echo "请以root用户运行此脚本（sudo ./install_dependencies.sh）"
    exit 1
fi

# 步骤1：更新系统
echo ""
echo "========== 步骤1：更新系统 =========="
apt update && apt upgrade -y

# 步骤2：安装ROS Noetic基础依赖
echo ""
echo "========== 步骤2：安装ROS Noetic =========="
apt install -y ros-noetic-desktop-full

# 步骤3：安装Gazebo和rotors_simulator依赖
echo ""
echo "========== 步骤3：安装Gazebo和rotors_simulator =========="
apt install -y ros-noetic-gazebo-ros ros-noetic-gazebo-ros-pkgs
apt install -y ros-noetic-rotors-simulator ros-noetic-mavros ros-noetic-mavros-extras
apt install -y ros-noetic-octomap-ros ros-noetic-joy

# 步骤4：安装Python依赖
echo ""
echo "========== 步骤4：安装Python依赖 =========="
pip install numpy scipy matplotlib opencv-python

# 步骤5：创建工作空间并编译
echo ""
echo "========== 步骤5：创建ROS工作空间 =========="
mkdir -p ~/drone_ws/src
cd ~/drone_ws/src

# 克隆rotors_simulator
if [ ! -d "rotors_simulator" ]; then
    git clone https://github.com/ethz-asl/rotors_simulator.git
fi

# 克隆mav_comm
if [ ! -d "mav_comm" ]; then
    git clone https://github.com/ethz-asl/mav_comm.git
fi

# 克隆glog_catkin
if [ ! -d "glog_catkin" ]; then
    git clone https://github.com/ethz-asl/glog_catkin.git
fi

# 克隆catkin_simple
if [ ! -d "catkin_simple" ]; then
    git clone https://github.com/catkin/catkin_simple.git
fi

# 编译工作空间
cd ~/drone_ws
catkin_make -DCMAKE_BUILD_TYPE=Release

# 步骤6：配置环境变量
echo ""
echo "========== 步骤6：配置环境变量 =========="
echo "" >> ~/.bashrc
echo "# 浙大高飞无人机团队 - ROS环境配置" >> ~/.bashrc
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
echo "source ~/drone_ws/devel/setup.bash" >> ~/.bashrc

echo ""
echo "========== 安装完成！=========="
echo "请执行以下命令使环境生效："
echo "source ~/.bashrc"
echo ""
echo "测试命令："
echo "roslaunch rotors_gazebo mav_hovering_example.launch"
echo ""
echo "如果Gazebo启动成功，说明环境配置正确！"