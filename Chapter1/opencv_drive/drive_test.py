# 导入游戏引擎库
import gym
import gym_donkeycar
import numpy as np
import cv2

# 设置模拟器环境，选择赛道
env = gym.make("donkey-generated-roads-v0")
# 重置当前场景
obv = env.reset()

# 运行100帧
for t in range(100):
    # 定义控制动作
    action = np.array([0.3, 0.5])  # 0.3表示转向，0.5表示油门
    # 执行动作
    img, reward, done, info = env.step(action)
    # 取一张图像保存
    if t == 20:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test.jpg', img)

# 运行完以后重置当前场景
obv = env.reset()