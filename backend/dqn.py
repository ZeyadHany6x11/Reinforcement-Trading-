import numpy as np
import random
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DQNNetwork(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_size: int, output_size: int):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size: int, n_actions: int = 3, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.98, lr: float = 5e-4,
                 batch_size: int = 64, memory_size: int = 5000,
                 target_update_freq: int = 10, replay_every: int = 4):
        self.state_size = state_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq
        self.replay_every = replay_every
        self.steps = 0

        if TORCH_AVAILABLE:
            self.device = torch.device("cpu")
            self.policy_net = DQNNetwork(state_size, n_actions).to(self.device)
            self.target_net = DQNNetwork(state_size, n_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.q_table_fallback = {}

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, float(reward), next_state, done))
        self.steps += 1

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        if TORCH_AVAILABLE:
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_vals = self.policy_net(t)
                return int(q_vals.argmax().item())
        else:
            return np.random.randint(self.n_actions)

    def replay(self):
        if self.steps % self.replay_every != 0:
            return
        if len(self.memory) < self.batch_size:
            return
        if not TORCH_AVAILABLE:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            next_q[dones_t] = 0.0
            target_q = rewards_t + self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if (self.steps // self.replay_every) % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
