import cv2
import numpy as np
import gym
import gym_donkeycar

# 导入自定义库
from img_analysis import region_of_interest, average_lines, compute_steer_angle

# 设置模拟器环境
env = gym.make("donkey-generated-roads-v0")
obv = env.reset()

# 开始启动，并获取首帧图像
action = np.array([0, 0.2])  # 动作控制，第1个转向值，第2个油门值
frame, reward, done, info = env.step(action)

# 运行2000次动作
for t in range(2000):
    # 高斯滤波去噪
    frame = cv2.GaussianBlur(frame, (5, 5), 1)

    # 转换图像到HSV空间
    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # 特定颜色区域检测
    lower_blue = np.array([15, 40, 40])
    upper_blue = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    lower_blue = np.array([0, 0, 200])
    upper_blue = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 边缘检测
    yellow_edge = cv2.Canny(yellow_mask, 200, 400)
    white_edge = cv2.Canny(white_mask, 200, 400)

    # 感兴趣区域提取
    yellow_croped = region_of_interest(yellow_edge, color="yellow")
    white_croped = region_of_interest(white_edge, color="white")

    # 直线检测
    rho = 1  # 距离精度：1像素
    angle = np.pi / 180  # 角度精度：1度
    min_thr = 10  # 最少投票数
    white_lines = cv2.HoughLinesP(
        white_croped, rho, angle, min_thr, np.array([]), minLineLength=8, maxLineGap=8
    )
    yellow_lines = cv2.HoughLinesP(
        yellow_croped, rho, angle, min_thr, np.array([]), minLineLength=8, maxLineGap=8
    )

    # 小线段聚类
    yellow_lane = average_lines(frame, yellow_lines, direction="left")
    white_lane = average_lines(frame, white_lines, direction="right")

    # 计算转向角
    steering_angle = compute_steer_angle(yellow_lane, white_lane, height, width)
    print(steering_angle)
    action = np.array([steering_angle, 0.2])  # 油门值恒定

    # 执行动作并重新获取图像
    frame, reward, done, info = env.step(action)

# 运行结束后重置当前场景
obv = env.reset()