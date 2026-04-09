# # 导入gymnasium库，重命名为gym方便使用
# import gymnasium as gym
# # 创建冰冻湖环境，开启可视化（render_mode="human"）
# env = gym.make("FrozenLake-v1", render_mode="human")
# # 重置环境，让机器人回到起点
# state = env.reset()
# # 让机器人随机走一步
# env.step(env.action_space.sample())
# # 渲染游戏窗口
# env.render()
# # 关闭环境（避免窗口卡死）
# env.close()
# # 打印成功提示
# print("环境配置成功！可以开始训练了～")
# ======================================
# 1. 导入所需库
# ======================================
import numpy as np  # 数值计算，用来创建和更新Q表
import gymnasium as gym  # 强化学习环境
import matplotlib.pyplot as plt  # 画图，展示训练效果
import time  # 可选，用来控制可视化速度

# 解决matplotlib画图的中文显示问题（Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# ======================================
# 2. 初始化环境和超参数（关键参数全解释，小白不用改）
# ======================================
# 创建冰冻湖环境：关闭打滑（is_slippery=False，动作和结果一一对应，降低入门难度）、4x4网格
env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4", render_mode="human")

# 超参数（大白话解释）
ALPHA = 0.8  # 学习率：机器人学新经验的速度，0.8=80%听新经验，20%保留旧经验
GAMMA = 0.95  # 折扣因子：机器人看重未来奖励的程度，0.95=很看重未来（要走长远路到终点）
EPSILON = 0.1  # 探索率：90%走已知最优路（利用），10%随机走（探索），避免钻牛角尖
EPISODES = 100  # 训练回合数：机器人走10000次冰冻湖，练得多学得会
MAX_STEPS = 100  # 每回合最大步数：防止机器人无限绕圈
ENABLE_RENDER = False  # 开启可视化：训练时能看到机器人移动（False=快速训练，无窗口）

# 初始化Q表：行数=状态数（16个，4x4网格），列数=动作数（4个，上下左右），初始值全为0
q_table = np.zeros((env.observation_space.n, env.action_space.n))
# 记录每回合的总奖励，用来画训练效果曲线
rewards_per_episode = []

# ======================================
# 3. 定义动作选择函数（ε-greedy策略：平衡探索和利用）
# ======================================
def choose_action(state):
    """
    根据ε-greedy策略给机器人选动作
    :param state: 机器人当前的位置（状态）
    :return: 选好的动作（0/1/2/3）
    """
    # 生成0-1的随机数，小于EPSILON（10%）则随机探索（走新路径）
    if np.random.uniform(0, 1) < EPSILON:
        return env.action_space.sample()  # 随机选一个动作（0/1/2/3）
    # 90%的情况利用已知经验：选Q表中当前状态下数值最大的动作
    else:
        # 兼容Gym新版本：如果状态是元组，提取纯数字（小白不用管，直接用）
        state = state[0] if isinstance(state, (tuple, list)) else state
        return np.argmax(q_table[state, :])  # 选Q值最大的动作

# ======================================
# 4. 开始训练机器人（核心步骤，逐行解释）
# ======================================
for episode in range(EPISODES):
    # 每回合开始：重置环境，让机器人回到起点
    state = env.reset()
    # 统一状态格式：提取纯数字（兼容Gym新版本）
    state = state[0] if isinstance(state, (tuple, list)) else state
    total_reward = 0  # 记录当前回合的总奖励
    done = False  # 标记：是否到达终点/掉入冰洞（True=回合结束）
    step = 0  # 记录当前回合走的步数

    # 每回合的移动循环：没结束且没超步数，就一直走
    while not done and step < MAX_STEPS:
        # 步骤1：根据当前位置选动作
        action = choose_action(state)
        # 步骤2：执行动作，获取环境反馈（新位置、奖励、是否结束等）
        next_state, reward, done, truncated, _ = env.step(action)
        # 步骤3：可视化窗口（可选，慢但能看到过程）
        if ENABLE_RENDER:
            env.render()
            time.sleep(0.05)  # 控制移动速度，避免画面闪太快（可删）
        # 步骤4：统一新状态格式（兼容Gym新版本）
        next_state = next_state[0] if isinstance(next_state, (tuple, list)) else next_state

        # 步骤5：核心！更新Q表（机器人记笔记，不用懂公式，直接用）
        target_q = reward + GAMMA * np.max(q_table[next_state, :])  # 理想的Q值（当前奖励+未来最大奖励）
        q_error = target_q - q_table[state, action]  # 实际Q值和理想Q值的误差
        q_table[state, action] += ALPHA * q_error  # 按误差更新Q值，学新经验

        # 步骤6：状态更新：机器人走到新位置，步数+1，奖励累加
        state = next_state
        step += 1
        total_reward += reward

    # 每回合结束：记录总奖励，方便后续画图
    rewards_per_episode.append(total_reward)

    # 每500回合打印一次训练进度，看机器人学得怎么样
    if (episode + 1) % 500 == 0:
        avg_reward = np.mean(rewards_per_episode[-500:])  # 最近500回合的平均奖励
        print(f"训练进度：{episode + 1}/{EPISODES}回合 | 最近500回合平均奖励：{avg_reward:.4f}")

# ======================================
# 5. 训练结果可视化：画曲线看机器人的学习进步
# ======================================
# 计算滑动平均奖励（窗口50），让曲线更平滑，小白能看清趋势
window_size = 50
smoothed_rewards = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(smoothed_rewards, color='blue')
plt.title("Q-Learning训练效果：滑动平均奖励曲线（窗口50）")
plt.xlabel("训练回合数")
plt.ylabel("平均奖励")
plt.grid(True)  # 显示网格，方便看
plt.show()

# ======================================
# 6. 测试训练好的机器人（关闭探索，只走最优路）
# ======================================
print("\n========== 开始测试训练好的机器人 ==========")
EPSILON = 0  # 探索率设为0，机器人只走Q表中最优的路，不探索
test_episodes = 10  # 测试10次，看成功率
success_count = 0  # 记录成功走到终点的次数

for episode in range(test_episodes):
    state = env.reset()
    state = state[0] if isinstance(state, (tuple, list)) else state
    done = False
    step = 0
    print(f"\n测试第{episode + 1}次：")
    while not done and step < MAX_STEPS:
        action = choose_action(state)  # 选最优动作
        next_state, reward, done, truncated, _ = env.step(action)
        env.render()
        time.sleep(0.2)  # 测试时慢一点，方便看
        next_state = next_state[0] if isinstance(next_state, (tuple, list)) else next_state
        # 打印每一步的细节，看机器人怎么走
        print(f"  第{step}步：位置={state} → 动作={action} → 新位置={next_state} → 奖励={reward}")
        state = next_state
        step += 1
    # 判断是否成功
    if done and reward == 1:
        success_count += 1
        print("  ✅ 成功走到终点！")
    else:
        print("  ❌ 掉入冰洞，失败！")

# 打印测试结果
success_rate = success_count / test_episodes * 100
print(f"\n测试结果：{test_episodes}次中成功{success_count}次 | 成功率：{success_rate}%")

# ======================================
# 7. 打印最终Q表（机器人的学习笔记本，看它学了什么）
# ======================================
print("\n========== 机器人的最终学习笔记（Q表） ==========")
print("行：机器人位置（0-15，4x4网格） | 列：动作（0=左、1=下、2=右、3=上） | 数值：动作好坏程度")
print(q_table)

# 关闭环境，结束程序
env.close()