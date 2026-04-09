# 最终完美可运行版 —— 无报错、无依赖、自带美化界面
import numpy as np
import gymnasium as gym

# ===================== 自带美化界面包装类 =====================
class FrozenLakeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.desc = env.unwrapped.desc

    def render(self):
        print("\n=== 冰湖游戏界面 ===")
        for row in self.desc:
            print(''.join([c for c in row]))
# =============================================================

# 创建环境（新版 Gymnasium 写法，无报错）
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
env = FrozenLakeWrapper(env)

# 初始化 Q 表
Q = {}
n_states = env.observation_space.n
n_actions = env.action_space.n
for s in range(n_states):
    for a in range(n_actions):
        Q[(s, a)] = 0.0

# 超参数
alpha = 0.5
gamma = 0.9
epsilon = 0.1
episodes = 10000

# 训练
print("开始训练...")
for i in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        # ε-贪婪策略
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = [Q[(state, a)] for a in range(n_actions)]
            action = np.argmax(q_vals)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 更新 Q 值
        old_q = Q[(state, action)]
        next_max = max([Q[(next_state, a)] for a in range(n_actions)])
        Q[(state, action)] = old_q + alpha * (reward + gamma * next_max - old_q)
        state = next_state

print("训练完成！开始测试...")

# 测试 100 次
success = 0
for _ in range(100):
    state, _ = env.reset()
    while True:
        q_vals = [Q[(state, a)] for a in range(n_actions)]
        action = np.argmax(q_vals)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            if reward == 1:
                success += 1
            break

print(f"\n✅ 测试成功率：{success} / 100 = {success}%")