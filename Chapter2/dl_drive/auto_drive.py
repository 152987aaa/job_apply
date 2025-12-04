# 导入系统库
import cv2
import numpy as np
import gym
import gym_donkeycar
import paddle

# 导入自定义库
from model import AutoDriveNet


# 设置模拟器环境
env = gym.make("donkey-generated-roads-v0")

# 重置当前场景
obv = env.reset()

# 设置GPU环境
paddle.set_device("cpu")

# 加载训练好的模型
model = AutoDriveNet()
checkpoint = paddle.load("D:\\Project_show_job\\Chapter2\\results\\model.pdparams")
model.set_state_dict(checkpoint)
model.eval()

# 开始启动
action = np.array([0, 0.2])  # 动作控制，第1个转向值，第2个油门值

# 执行动作并获取图像
frame, reward, done, info = env.step(action)

# 运行2500次动作
for t in range(2500):
    # 图像转Tensor
    img = paddle.to_tensor(frame.copy(), stop_gradient=True)
    # 归一化到0~1
    img /= 255.0
    # 调整通道，从HWC调整为CHW
    img = img.transpose([2, 0, 1])
    # 扩充维度，从CHW扩充为NCHW
    img.unsqueeze_(0)
    # 模型推理
    with paddle.no_grad():
        # 前向推理获得预测的转向角度
        prelabel = model(img).squeeze(0).cpu().detach().numpy()
        steering_angle = prelabel[0]
        # 执行动作并重新获取图像
        factor = 1.5  # 动作增强因子
        action = np.array([steering_angle * factor, 0.2])
        frame, reward, done, info = env.step(action)

# 运行完以后重置当前场景
obv = env.reset()
# # 导入系统库
# import cv2
# import numpy as np
# import gym
# import gym_donkeycar
# import paddle
# from sklearn.externals.array_api_compat.cupy import acosh
# from sympy import factor
#
# from Chapter1.opencv_drive.auto_drive import steering_angle
# #导入自定义库
# from model import AutoDriveNet
# #设置模拟器环境
# env = gym.make('donkey-generated-roads-v0')
# #重置当前场景
# obv = env.reset()
# #设置GPU环境
# paddle.set_device("gpu")
# #加载训练好的模型
# model = AutoDriveNet()
# checkpoint = paddle.load("./result/model.pdparams")
# model.set_state_dict(checkpoint)
# model.eval()
#
# #开始启动
# action = np.array([0,0.2])
# frame,reword,done,info = env.step(action)
# #执行2500次动作
# for t in range(2500):
#     #图像转Tensor
#     img = paddle.to_tensor(frame.copy(),stop_gradient=True)
#     #归一化
#     img/=255.0
#     #调整通道，从HWC调整为CHW
#     img = img.transpose([2,0,1])
#     #扩充维度
#     img.unsqueeze_(0)
#     #模型推理
#     with paddle.no_grad():
#         prelabel = model(img).squeeze(0).cpu().detach().numpy()
#         steering_angle = prelabel[0]
#         #执行动作并重新获取图像
#         factor=1.5
#         action = np.array([steering_angle*factor,0.2])
#         frame,reword,done,info = env.step(action)
# obv = env.reset()
