import numpy as np
from collections import defaultdict


class QLearningAgent:
    def __init__(self, state_size: int, n_actions: int = 3, alpha: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.98, n_bins: int = 5):
        self.state_size = state_size
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_bins = n_bins
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.bins = np.linspace(-0.3, 0.3, n_bins - 1)

    def _discretize(self, state: np.ndarray) -> tuple:
        price_part = state[:-1]
        position = int(state[-1])
        binned = tuple(np.digitize(s, self.bins) for s in price_part)
        return binned + (position,)

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        key = self._discretize(state)
        return int(np.argmax(self.q_table[key]))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        s = self._discretize(state)
        ns = self._discretize(next_state)
        best_next = float(np.max(self.q_table[ns]))
        td_target = float(reward) + self.gamma * best_next
        td_error = td_target - float(self.q_table[s][action])
        self.q_table[s][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
