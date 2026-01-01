import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

# 1. THE REWARD-ADJUSTED ENVIRONMENT
class SentimentEnv(gym.Env):
    """
    Custom environment where correctly identifying rare classes 
    (Bullish/Bearish) gives higher rewards than Neutral ones.
    """
    def __init__(self, features, labels, class_weights):
        super(SentimentEnv, self).__init__()
        self.features = features
        self.labels = labels
        self.class_weights = class_weights # Higher values for rare classes
        self.current_step = 0
        
        # Action space: 0: Bearish, 1: Neutral, 2: Bullish
        self.action_space = gym.spaces.Discrete(3)
        # Observation space: The text feature vector size
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(features.shape[1],), dtype=np.float32)

    def step(self, action):
        target = self.labels[self.current_step]
        
        # Reward Logic for Class Imbalance
        if action == target:
            reward = 1.0 * self.class_weights[target] # Scaled by rarity
        else:
            reward = -1.0 # Penalty for wrong classification
            
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        obs = self.features[self.current_step]
        return obs, reward, done, False, {}

    def reset(self, seed=None):
        self.current_step = 0
        return self.features[self.current_step], {}

# 2. DILATED CNN + ACTOR-CRITIC MODEL
class DilatedActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DilatedActorCritic, self).__init__()
        # Dilated CNN layers to capture long-range text dependencies
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, dilation=4, padding=4),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flat size after convolutions
        self.fc = nn.Linear(128 * input_dim, 256)
        
        self.actor = nn.Linear(256, action_dim)   # Policy head
        self.critic = nn.Linear(256, 1)          # Value head

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dim for Conv1d
        x = self.feature_extractor(x)
        x = torch.relu(self.fc(x))
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)

# 3. THE OFF-POLICY PPO TRAINER
class OffPolicyPPO:
    def __init__(self, model, lr=3e-4, epsilon=0.2, gamma=0.99):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.eps = epsilon
        self.gamma = gamma
        self.buffer = deque(maxlen=2000) # The "Off-Policy" Replay Buffer

    def store(self, transition):
        self.buffer.append(transition)

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size: return
        
        # Sample from buffer (Off-policy sampling)
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, log_p_old = zip(*batch)
        
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(a).unsqueeze(1)
        r = torch.tensor(r).unsqueeze(1)
        log_p_old = torch.tensor(log_p_old).unsqueeze(1)

        # PPO Clipped Objective
        probs, values = self.model(s)
        curr_log_p = torch.log(probs.gather(1, a))
        ratio = torch.exp(curr_log_p - log_p_old)
        
        advantage = r - values.detach() # Simplified Advantage
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
        
        loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(values, r)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()