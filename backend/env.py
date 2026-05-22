import numpy as np


class TradingEnv:
    def __init__(self, prices: np.ndarray, window_size: int = 10, initial_balance: float = 10000.0):
        self.prices = prices
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.buy_price = 0.0
        self.prev_net_worth = self.initial_balance
        return self._get_state()

    def _get_state(self):
        window = self.prices[self.current_step - self.window_size: self.current_step]
        norm = window / window[0] - 1.0
        position = np.array([1.0 if self.shares_held > 0 else 0.0], dtype=np.float32)
        return np.concatenate([norm.astype(np.float32), position])

    def step(self, action: int):
        current_price = float(self.prices[self.current_step])
        reward = 0.0
        invalid_penalty = -0.1

        if action == 1:  
            if self.shares_held == 0 and self.balance >= current_price:
                shares_to_buy = int(self.balance // current_price)
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
                self.buy_price = current_price
            else:
                reward += invalid_penalty

        elif action == 2:  
            if self.shares_held > 0:
                revenue = self.shares_held * current_price
                realized_pnl = (current_price - self.buy_price) / self.buy_price if self.buy_price > 0 else 0.0
                reward += float(realized_pnl) * 10.0  
                self.balance += revenue
                self.shares_held = 0
                self.buy_price = 0.0
            else:
                reward += invalid_penalty

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        net_worth = self.portfolio_value()
        step_reward = (net_worth - self.prev_net_worth) / self.prev_net_worth
        reward += float(step_reward) * 5.0
        self.prev_net_worth = net_worth

        next_state = self._get_state() if not done else np.zeros(self.window_size + 1, dtype=np.float32)
        return next_state, reward, done

    def portfolio_value(self):
        return float(self.balance) + self.shares_held * float(self.prices[self.current_step - 1])
