#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浙大高飞无人机团队 | 强化学习路径规划模块
教学目标：掌握DQN、PPO、DDPG在路径规划中的应用
对应论文：
- PPO: 《Proximal Policy Optimization Algorithms》
- DDPG: 《Continuous Control with Deep Reinforcement Learning》
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DQNAgent:
    """DQN深度Q网络智能体"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        """
        初始化DQN智能体
        :param state_size: 状态维度
        :param action_size: 动作数量
        :param learning_rate: 学习率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # 经验回放缓冲区
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 32
        
        # 折扣因子
        self.gamma = 0.99
        
        # ε-greedy参数
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # 创建神经网络
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
    
    def build_model(self):
        """构建Q网络"""
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(self.state_size,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """更新目标网络"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            
            target = self.model.predict(state, verbose=0)
            
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.model.fit(state, target, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class PPOAgent:
    """PPO近端策略优化智能体"""
    
    def __init__(self, state_size, action_size, learning_rate=0.0003):
        """
        初始化PPO智能体
        :param state_size: 状态维度
        :param action_size: 动作维度（连续动作）
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # PPO参数
        self.gamma = 0.99
        self.epsilon_clip = 0.2
        self.critic_weight = 0.5
        self.entropy_weight = 0.01
        
        # 创建策略网络和价值网络
        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()
        
        # 优化器
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
    
    def build_actor(self):
        """构建策略网络"""
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        
        # 均值输出
        mu = layers.Dense(self.action_size, activation='tanh')(x)
        
        # 标准差输出（取指数确保非负）
        log_std = layers.Dense(self.action_size)(x)
        std = layers.Lambda(lambda x: tf.exp(x))(log_std)
        
        return models.Model(inputs=inputs, outputs=[mu, std])
    
    def build_critic(self):
        """构建价值网络"""
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(self.state_size,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        return model
    
    def act(self, state):
        """选择动作（带探索）"""
        state = np.reshape(state, [1, self.state_size])
        mu, std = self.actor_model.predict(state, verbose=0)
        
        # 从正态分布采样动作
        action = mu + std * np.random.randn(*mu.shape)
        action = np.clip(action, -1, 1)
        
        # 计算对数概率
        log_prob = -0.5 * ((action - mu) / std)**2 - np.log(std) - 0.5 * np.log(2 * np.pi)
        log_prob = np.sum(log_prob)
        
        return action[0], log_prob
    
    def compute_returns(self, rewards, dones, next_value):
        """计算折扣回报"""
        returns = []
        running_sum = next_value
        
        for t in reversed(range(len(rewards))):
            running_sum = rewards[t] + self.gamma * running_sum * (1 - dones[t])
            returns.insert(0, running_sum)
        
        # 标准化回报
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        return returns
    
    def train(self, states, actions, log_probs, returns):
        """训练PPO"""
        with tf.GradientTape() as tape:
            # 获取当前策略的均值和标准差
            mu, std = self.actor_model(states)
            
            # 计算新的对数概率
            action_dist = tf.distributions.Normal(mu, std)
            new_log_probs = action_dist.log_prob(actions)
            new_log_probs = tf.reduce_sum(new_log_probs, axis=1)
            
            # 计算优势
            values = self.critic_model(states)
            advantages = returns - tf.squeeze(values)
            
            # 计算策略损失（clip PPO）
            ratio = tf.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # 计算价值损失
            value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))
            
            # 计算熵损失（鼓励探索）
            entropy = tf.reduce_mean(action_dist.entropy())
            entropy_loss = -self.entropy_weight * entropy
            
            # 总损失
            total_loss = policy_loss + self.critic_weight * value_loss + entropy_loss
        
        # 反向传播
        grads = tape.gradient(total_loss, 
            self.actor_model.trainable_variables + self.critic_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, 
            self.actor_model.trainable_variables + self.critic_model.trainable_variables))

class NavigationEnv:
    """导航环境"""
    
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.agent_pos = np.array([1, 1])
        self.goal_pos = np.array([grid_size-2, grid_size-2])
        
        # 创建障碍物
        self.obstacles = set()
        for i in range(5, 8):
            for j in range(5, 15):
                self.obstacles.add((i, j))
        for i in range(12, 15):
            for j in range(5, 10):
                self.obstacles.add((i, j))
    
    def reset(self):
        """重置环境"""
        self.agent_pos = np.array([1, 1])
        return self.get_state()
    
    def get_state(self):
        """获取状态（距离目标的相对坐标）"""
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.goal_pos[0] - self.agent_pos[0],
            self.goal_pos[1] - self.agent_pos[1]
        ])
    
    def step(self, action):
        """执行动作"""
        # 动作：0=上, 1=下, 2=左, 3=右
        directions = [np.array([0, 1]), np.array([0, -1]), 
                     np.array([-1, 0]), np.array([1, 0])]
        
        self.agent_pos += directions[action]
        
        # 边界检查
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size-1)
        
        # 障碍物检查
        if tuple(self.agent_pos) in self.obstacles:
            self.agent_pos -= directions[action]
            reward = -10
            done = False
        elif np.all(self.agent_pos == self.goal_pos):
            reward = 100
            done = True
        else:
            # 距离奖励（越近奖励越高）
            dist = np.linalg.norm(self.agent_pos - self.goal_pos)
            reward = -dist / self.grid_size
            done = False
        
        return self.get_state(), reward, done

def train_dqn():
    """训练DQN导航智能体"""
    env = NavigationEnv()
    state_size = 4
    action_size = 4
    
    agent = DQNAgent(state_size, action_size)
    
    episodes = 500
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
                break
        
        agent.replay()
    
    print("DQN训练完成！")

if __name__ == '__main__':
    train_dqn()