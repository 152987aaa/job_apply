import cv2
import numpy as np
import math


def region_of_interest(edges, color="yellow"):
    '''提取感兴趣区域'''
    height, width = edges.shape
    mask = np.zeros_like(edges)
    # 定义感兴趣区域掩码轮廓
    if color == "yellow":
        polygon = np.array(
            [
                [
                    (0, height * 1 / 2),
                    (width * 1 / 2, height * 1 / 2),
                    (width * 1 / 2, height),
                    (0, height),
                ]
            ],
            np.int32,
        )
    else:
        polygon = np.array(
            [
                [
                    (width * 1 / 2, height * 1 / 2),
                    (width, height * 1 / 2),
                    (width, height),
                    (width * 1 / 2, height),
                ]
            ],
            np.int32,
        )
    # 填充感兴趣区域掩码
    cv2.fillPoly(mask, polygon, 255)
    # 提取感兴趣区域
    croped_edge = cv2.bitwise_and(edges, mask)
    return croped_edge


def make_points(frame, line):
    """根据直线斜率和截距计算指定高度处的起始坐标"""
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 1 / 2)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]


def average_lines(frame, lines, direction="left"):
    """对小线段进行聚类"""
    lane_line = []
    if lines is None:
        print(direction + "没有检测到线段")
        return lane_line
    fits = []
    # 计算每条小线段的斜率和截距
    for line in lines:
        for x1, y1, x2, y2 in line:
            # 最小二乘法拟合
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]  # 斜率
            intercept = fit[1]  # 截距
            if direction == "left" and slope < 0:
                fits.append((slope, intercept))
            elif direction == "right" and slope > 0:
                fits.append((slope, intercept))
    # 计算所有小线段的平均斜率和截距
    if len(fits) > 0:
        fit_average = np.average(fits, axis=0)
        lane_line = make_points(frame, fit_average)
    return lane_line


def display_line(frame, line, line_color=(0, 0, 255), line_width=3):
    """在原图上合成展示线段"""
    line_img = np.zeros_like(frame)
    x1, y1, x2, y2 = line
    cv2.line(line_img, (x1, y1), (x2, y2), line_color, line_width)
    line_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return line_img


def compute_steer_angle(yellow_lane, white_lane, height, width):
    '''计算转向角'''
    x_offset = 0
    y_offset = int(height / 2)
    # 分情况计算横纵偏移值
    if len(yellow_lane) > 0 and len(white_lane) > 0:  # 检测到2条线
        _, _, left_x2, _ = yellow_lane
        _, _, right_x2, _ = white_lane
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
    elif len(yellow_lane) > 0:  # 只检测到黄色车道线
        x1, _, x2, _ = yellow_lane
        x_offset = x2 - x1
    elif len(white_lane) > 0:  # 只检测到白色车道线
        x1, _, x2, _ = white_lane
        x_offset = x2 - x1
    else:  # 都没检测到
        print("检测不到车道线，即将停止")
        return -1
    # 计算最终转向角度
    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # 转换为角度
    steering_angle = angle_to_mid_deg / 45.0  # 归一化到区间
    return steering_angle