import cv2
import numpy as np
import math

from Chapter1.opencv_drive.auto_drive import height

# ----------------------1.基于HSV空间的特定颜色区域提取------------------
# 读取图像并转换到HSV空间
frame = cv2.imread("test.jpg")
frame = cv2.GaussianBlur(frame, (5, 5), 1)  # -----2.基于高斯模糊的噪声滤除----
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# 黄色线检测
lower_blue = np.array([15, 40, 40])
upper_blue = np.array([45, 255, 255])
yellow_mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imwrite("yellow_mask.jpg", yellow_mask)
# 白色线检测
lower_blue = np.array([0, 0, 200])
upper_blue = np.array([180, 30, 255])
white_mask = cv2.inRange(hsv, lower_blue, upper_blue)
# 保存中间结果
cv2.imwrite("yellow_mask.jpg", yellow_mask)
cv2.imwrite("white_mask.jpg", white_mask)

# -------------3基于canny算法边缘线检测-------------------
# 黄色边缘线提取
yellow_edge = cv2.Canny(yellow_mask, 200, 400)
# 白色边缘线提取
white_edge = cv2.Canny(white_mask, 200, 400)
# 保存中间结果
cv2.imwrite("yellow_edge.jpg", yellow_edge)
cv2.imwrite("white_edge.jpg", white_edge)

# -----------4感兴趣区域提取----------------------
def region_of_interest(edges, color="yellow"):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    # 定义感兴趣区域掩码轮廓
    if color == "yellow":
        polygon = np.array([[(0, height*1/2),
                             (width*1/2, height*1/2),
                             (width *1/2, height),
                             (0,height)]],np.int32)
    else:
        polygon = np.array([[(width*1/2, height*1/2),
                             (width,height*1/2),
                             (width, height),
                             (width*1/2, height)]],np.int32)
    # 填充感兴趣区域掩码
    cv2.fillPoly(mask, polygon, 255)
    # 提取感兴趣区域
    cropped_edge = cv2.bitwise_and(edges, mask)
    return cropped_edge

# 黄色车道线感兴趣区域提取
yellow_croped = region_of_interest(yellow_edge, color="yellow")
cv2.imwrite("yellow_cropped.jpg", yellow_croped)
white_croped = region_of_interest(white_edge, color="white")
cv2.imwrite("white_cropped.jpg", white_croped)

# ------------5基于霍夫变换的直线检测----------
rho = 1 #距离精度：1像素
angle = np.pi/180 # 角度精度1°
min_thr = 10 #最少投票数
white_lines = cv2.HoughLinesP(white_croped, rho, angle, min_thr,np.array([]), minLineLength=8, maxLineGap=8)
yellow_lines = cv2.HoughLinesP(yellow_croped, rho, angle, min_thr, np.array([]), minLineLength=8,maxLineGap=8)
#输出查看返回的线段
print(white_lines)

import cv2
import numpy as np

# ---------------------- 可视化检测到的白色线段 ----------------------
# 1. 读取ROI后的图像（white_cropped.jpg），作为画图的背景
img = cv2.imread("white_cropped.jpg")  # 读取白色ROI图像
# 如果img是单通道（黑白图），转成三通道（彩色图），方便用彩色画线段
if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 2. 遍历white_lines中的每条线段，画在图像上
if white_lines is not None:  # 确保检测到了线段
    for line in white_lines:
        x1, y1, x2, y2 = line[0]  # 取出每条线段的两个端点
        # 画线段：颜色(0,255,0)=绿色，线宽2像素，线条类型为实线
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        # 给每个端点画红色小圆点（方便看端点位置）
        cv2.circle(img, (x1, y1), 3, (0, 0, 255), -1)  # 端点1：红色圆点
        cv2.circle(img, (x2, y2), 3, (0, 0, 255), -1)  # 端点2：红色圆点

# 3. 保存并显示结果图
cv2.imwrite("white_lines_visualize.jpg", img)  # 保存标注后的图
cv2.imshow("White Lines Detection", img)  # 弹出窗口显示
cv2.waitKey(0)  # 按任意键关闭窗口
cv2.destroyAllWindows()

# --------小线段聚类--------------
def make_points(frame,line):
    """根据直线斜率和截距计算指定高度处的起始坐标"""
    height,width,_ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1*1/2)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [x1,y1,x2,y2]
def average_lines(frame,lines,direction="left"):
    """对小线段进行聚类"""
    lane_line=[]
    if lines is None:
        print(direction+"没有检测到线段")
        return  lane_line
    fits = []
    #计算每条小线段的斜率和截距
    for line in lines:
        for x1,y1,x2,y2 in line:
            #最小二乘法拟合
            fit = np.polyfit((x1,x2),(y1,y2),1)
            slope = fit[0]
            intercept = fit[1]
            if direction == "left" and slope <0:
                fits.append((slope,intercept))
            elif direction == "right" and slope >0:
                fits.append((slope,intercept))
        #计算所有小线段的平均斜率和截距
    if len(fits)>0:
        fit_average = np.average(fits,axis=0)
        lane_line = make_points(frame, fit_average)
    return lane_line
# 聚合线段
yellow_lane = average_lines(frame,yellow_lines,direction="left")
white_lane = average_lines(frame,white_lines,direction="right")
print(white_lane)
print(yellow_lane)

#----------7可视化显示检测结果------------------
def display_line(frame, line, line_color=(0, 0, 255),line_width=3):
    """在原图上合成展示线段"""
    line_img = np.zeros_like(frame)
    x1,y1,x2,y2 = line
    cv2.line(line_img,(x1,y1),(x2,y2),line_color,line_width)
    line_img = cv2.addWeighted(frame, 0.8, line_img,1,1)
    return line_img
#显示检测结果
img_yellow = display_line(frame,yellow_lane,line_color=(0, 0, 255),line_width=3)
img_white = display_line(frame,white_lane,line_color=(0, 0, 255),line_width=3)
cv2.imwrite("img_yellow.jpg", img_yellow)
cv2.imwrite("img_white.jpg", img_white)

#-----------8动作控制---------------
def compute_steer_angle(yellow_lane,white_lane,height,width):
    x_offset = 0
    y_offset = int(height/2)
    #分情况计算横纵偏移值
    if len(yellow_lane) >0 and len(white_lane)>0:
        _,_,left_x2,_ = yellow_lane
        _,_,right_x2,_ = white_lane
        mid = int(width/2)
        x_offset = (left_x2+right_x2)/2 - mid
    elif len(yellow_lane)>0:
        x1,_,x2,_ = yellow_lane
        x_offset = x2-x1
    elif len(white_lane)>0:
        x1,_,x2,_ = white_lane
        x_offset = x2-x1
    else:
        print("检测不到车道线，即将停止")
        return -1
    # 计算最终转向角度
    angle_to_mid_radian = math.atan(x_offset/y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian*180.0/math.pi)
    steering_angle = angle_to_mid_deg/45.0#归一化到区间
    return steering_angle
#计算转向角
height,width,_ = frame.shape
steering_angle = compute_steer_angle(yellow_lane,white_lane,height,width)
print(steering_angle)
