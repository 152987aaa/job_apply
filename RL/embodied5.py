import numpy as np


def dh_matrix(theta, d, a, alpha):
    """标准DH矩阵"""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])


def forward_kinematics_UR5(q):
    """
    UR5 正运动学
    q: 6个关节角 [q1,q2,q3,q4,q5,q6]
    返回: 基坐标系到末端坐标系 T_06
    """
    # ------------- UR5 DH参数 ----------------
    alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
    a = np.array([0, -425.00, -392.25, 0, 0, 0])
    d = np.array([89.16, 0, 0, 109.15, 94.65, 82.30])
    theta = q

    # 逐次计算变换矩阵
    T_01 = dh_matrix(theta[0], d[0], a[0], alpha[0])
    T_12 = dh_matrix(theta[1], d[1], a[1], alpha[1])
    T_23 = dh_matrix(theta[2], d[2], a[2], alpha[2])
    T_34 = dh_matrix(theta[3], d[3], a[3], alpha[3])
    T_45 = dh_matrix(theta[4], d[4], a[4], alpha[4])
    T_56 = dh_matrix(theta[5], d[5], a[5], alpha[5])

    # 总变换：从基座到末端
    T_06 = T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56
    return T_06


# ===================== 测试零位 =====================
q_zero = [0, 0, 0, 0, 0, 0]
T = forward_kinematics_UR5(q_zero)
end_pos = T[:3, 3]  # 末端坐标 xyz

print("UR5 零位末端坐标 (mm):")
print(np.round(end_pos, 4))