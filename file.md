# Simple Explanation of `env.py`, `q_learning.py`, and `dqn.py`

This project contains a small reinforcement learning trading setup.

The main idea is:

1. `env.py` creates a simple stock trading environment.
2. `q_learning.py` creates a basic Q-Learning agent that learns using a table.
3. `dqn.py` creates a Deep Q-Network agent that learns using a neural network.

The agents try to learn when to:

- `0`: Hold
- `1`: Buy
- `2`: Sell

The goal is to make good trading decisions and increase portfolio value over time.

---

# 1. `env.py`

File path:

```text
backend/env.py
```

This file defines the trading environment. The environment is like the "game" that the reinforcement learning agent plays.

The agent observes prices, chooses an action, and receives a reward.

## Imports

```python
import numpy as np
```

This imports NumPy.

NumPy is used for:

- storing price data
- creating arrays
- doing numerical calculations

---

## Class: `TradingEnv`

```python
class TradingEnv:
```

This class represents the trading environment.

It stores:

- the stock prices
- the current time step
- the agent's cash balance
- how many shares the agent owns
- the previous portfolio value

The environment is responsible for:

- giving the agent the current state
- applying the agent's action
- calculating rewards
- deciding when the episode is finished

---

## Constructor: `__init__`

```python
def __init__(self, prices: np.ndarray, window_size: int = 10, initial_balance: float = 10000.0):
```

This function runs when a new `TradingEnv` object is created.

### Parameters

```python
prices: np.ndarray
```

This is the list or array of stock prices.

Example:

```python
[100, 101, 102, 99, 105]
```

The environment uses these prices one by one as time moves forward.

```python
window_size: int = 10
```

This means the agent sees the last 10 prices by default.

Instead of seeing only today's price, the agent sees a small history of prices.

For example, if `window_size = 10`, the state contains the last 10 price changes.

```python
initial_balance: float = 10000.0
```

This is the starting money.

By default, the agent starts with `10000.0`.

### Stored values

```python
self.prices = prices
self.window_size = window_size
self.initial_balance = initial_balance
self.reset()
```

The environment saves the prices, window size, and starting balance.

Then it calls:

```python
self.reset()
```

This prepares the environment for a new trading episode.

---

## Function: `reset`

```python
def reset(self):
```

This function resets the environment back to the beginning.

It is usually called before training starts, or before a new episode starts.

```python
self.current_step = self.window_size
```

The environment starts at `window_size`, not at `0`.

Why?

Because the state needs a window of previous prices.

If `window_size = 10`, the first state uses prices from index `0` to index `9`, and the current step starts at index `10`.

```python
self.balance = self.initial_balance
```

The cash balance is reset to the starting money.

```python
self.shares_held = 0
```

The agent starts with no shares.

```python
self.buy_price = 0.0
```

This stores the price where the agent bought shares.

At the start, the agent has not bought anything, so it is `0.0`.

```python
self.prev_net_worth = self.initial_balance
```

This stores the previous portfolio value.

At the beginning, portfolio value is just the starting cash.

```python
return self._get_state()
```

After resetting, the environment returns the first state.

The state is what the agent sees.

---

## Function: `_get_state`

```python
def _get_state(self):
```

This function creates the current state for the agent.

The state is the information given to the agent so it can choose an action.

```python
window = self.prices[self.current_step - self.window_size: self.current_step]
```

This takes the recent price window.

Example:

If:

```python
current_step = 10
window_size = 10
```

Then this takes prices from index `0` to index `9`.

If:

```python
current_step = 20
window_size = 10
```

Then this takes prices from index `10` to index `19`.

```python
norm = window / window[0] - 1.0
```

This normalizes the prices.

Instead of using raw prices like:

```text
100, 102, 105
```

It converts them into relative changes compared to the first price in the window.

Example:

If the window is:

```text
[100, 105, 110]
```

Then:

```text
100 / 100 - 1 = 0.00
105 / 100 - 1 = 0.05
110 / 100 - 1 = 0.10
```

So the normalized state becomes:

```text
[0.00, 0.05, 0.10]
```

This helps the agent understand price movement instead of depending on exact price values.

```python
position = np.array([1.0 if self.shares_held > 0 else 0.0], dtype=np.float32)
```

This tells the agent whether it currently owns shares.

- `1.0` means the agent is holding shares.
- `0.0` means the agent is not holding shares.

This is important because the best action depends on the current position.

For example:

- If the agent already owns shares, buying again may be invalid.
- If the agent owns no shares, selling is invalid.

```python
return np.concatenate([norm.astype(np.float32), position])
```

This combines the normalized price window and the position value into one state array.

If `window_size = 10`, then:

- 10 values come from price history
- 1 value shows whether shares are held

So the total state size is:

```text
window_size + 1
```

For the default settings:

```text
10 + 1 = 11
```

---

## Function: `step`

```python
def step(self, action: int):
```

This is the most important function in the environment.

It applies the action chosen by the agent.

The action can be:

- `0`: Hold
- `1`: Buy
- `2`: Sell

The function returns:

```python
next_state, reward, done
```

Where:

- `next_state` is the new state after the action
- `reward` tells the agent how good or bad the action was
- `done` tells whether the episode has ended

---

## Getting the Current Price

```python
current_price = float(self.prices[self.current_step])
```

This gets the price at the current time step.

The value is converted to `float`.

---

## Starting Reward

```python
reward = 0.0
invalid_penalty = -0.1
```

The reward starts at `0.0`.

If the agent tries an invalid action, it gets a small penalty of `-0.1`.

An invalid action means:

- trying to buy when already holding shares
- trying to sell when holding no shares
- trying to buy without enough money

---

## Action 1: Buy

```python
if action == 1:  # Buy
```

If the action is `1`, the agent wants to buy.

```python
if self.shares_held == 0 and self.balance >= current_price:
```

The agent can buy only if:

- it is not already holding shares
- it has enough money to buy at least one share

```python
shares_to_buy = int(self.balance // current_price)
```

This calculates how many whole shares the agent can buy.

The `//` operator means integer division.

Example:

If:

```text
balance = 10000
current_price = 300
```

Then:

```text
10000 // 300 = 33
```

So the agent buys `33` shares.

```python
self.shares_held += shares_to_buy
```

This adds the bought shares to the agent's holdings.

```python
self.balance -= shares_to_buy * current_price
```

This subtracts the cost from the cash balance.

```python
self.buy_price = current_price
```

This stores the price where the shares were bought.

It will be used later when calculating profit or loss after selling.

```python
else:
    reward += invalid_penalty
```

If the agent cannot buy, it receives the invalid action penalty.

---

## Action 2: Sell

```python
elif action == 2:  # Sell
```

If the action is `2`, the agent wants to sell.

```python
if self.shares_held > 0:
```

The agent can sell only if it owns shares.

```python
revenue = self.shares_held * current_price
```

This calculates how much money the agent gets from selling all shares.

```python
realized_pnl = (current_price - self.buy_price) / self.buy_price if self.buy_price > 0 else 0.0
```

This calculates the realized profit or loss percentage.

`pnl` means profit and loss.

Example:

If the agent bought at `100` and sold at `110`:

```text
(110 - 100) / 100 = 0.10
```

That means `10%` profit.

If the agent bought at `100` and sold at `90`:

```text
(90 - 100) / 100 = -0.10
```

That means `10%` loss.

```python
reward += float(realized_pnl) * 10.0
```

The profit or loss percentage is multiplied by `10`.

This makes the reward stronger.

Example:

If profit is `0.10`, then:

```text
0.10 * 10 = 1.0
```

So the reward gets `+1.0`.

If loss is `-0.10`, then:

```text
-0.10 * 10 = -1.0
```

So the reward gets `-1.0`.

```python
self.balance += revenue
```

The sale money is added to the cash balance.

```python
self.shares_held = 0
```

After selling, the agent no longer owns shares.

```python
self.buy_price = 0.0
```

The buy price is reset because there is no open position anymore.

```python
else:
    reward += invalid_penalty
```

If the agent tries to sell without shares, it receives the invalid action penalty.

---

## Action 0: Hold

There is no explicit `if action == 0` block.

That means if the action is `0`, the environment does not buy or sell.

It simply moves to the next time step.

The agent may still receive a reward based on whether the portfolio value changed.

---

## Moving Time Forward

```python
self.current_step += 1
```

After the action, the environment moves one step forward in time.

```python
done = self.current_step >= len(self.prices) - 1
```

This checks if the episode is finished.

The episode ends when the environment reaches near the end of the price data.

---

## Portfolio Value Reward

```python
net_worth = self.portfolio_value()
```

This calculates the current total portfolio value.

Portfolio value means:

```text
cash balance + value of currently held shares
```

```python
step_reward = (net_worth - self.prev_net_worth) / self.prev_net_worth
```

This calculates how much the portfolio value changed compared to the previous step.

Example:

If previous net worth was `10000` and current net worth is `10100`:

```text
(10100 - 10000) / 10000 = 0.01
```

That means the portfolio increased by `1%`.

```python
reward += float(step_reward) * 5.0
```

The step reward is multiplied by `5`.

This gives the agent extra reward when the portfolio value increases, and negative reward when it decreases.

```python
self.prev_net_worth = net_worth
```

The current net worth becomes the previous net worth for the next step.

---

## Returning the Next State

```python
next_state = self._get_state() if not done else np.zeros(self.window_size + 1, dtype=np.float32)
```

If the episode is not done, the environment returns the next real state.

If the episode is done, it returns an array of zeros.

The zero array has the same size as a normal state:

```text
window_size + 1
```

```python
return next_state, reward, done
```

The function returns:

- the next state
- the reward
- whether the episode ended

---

## Function: `portfolio_value`

```python
def portfolio_value(self):
```

This calculates the total value of the agent's portfolio.

```python
return float(self.balance) + self.shares_held * float(self.prices[self.current_step - 1])
```

The portfolio value is:

```text
cash balance + current value of held shares
```

It uses:

```python
self.current_step - 1
```

because `current_step` is increased before the reward is calculated.

So the latest used trading price is at `current_step - 1`.

---

# 2. `q_learning.py`

File path:

```text
backend/q_learning.py
```

This file defines a simple Q-Learning agent.

Q-Learning is a reinforcement learning method where the agent learns a table of values.

The table tells the agent:

```text
For this state, how good is each action?
```

The table is called a Q-table.

---

## Imports

```python
import numpy as np
from collections import defaultdict
```

`numpy` is used for numerical calculations.

`defaultdict` is used to create Q-table rows automatically.

If the agent sees a new state, `defaultdict` creates default action values for it.

---

## Class: `QLearningAgent`

```python
class QLearningAgent:
```

This class represents the table-based learning agent.

It learns by storing Q-values in a dictionary.

Each state gets a list of action values.

Example:

```text
state A -> [value for hold, value for buy, value for sell]
```

---

## Constructor: `__init__`

```python
def __init__(self, state_size: int, n_actions: int = 3, alpha: float = 0.1,
             gamma: float = 0.95, epsilon: float = 1.0, epsilon_min: float = 0.05,
             epsilon_decay: float = 0.98, n_bins: int = 5):
```

This function sets up the Q-Learning agent.

### Parameters

```python
state_size: int
```

The size of the state.

In this project, the state size is usually:

```text
window_size + 1
```

The `+1` is for the position value.

```python
n_actions: int = 3
```

The number of possible actions.

The default is `3`:

- hold
- buy
- sell

```python
alpha: float = 0.1
```

This is the learning rate.

It controls how much the agent updates its Q-values after each experience.

A higher `alpha` means the agent changes its values faster.

A lower `alpha` means the agent learns more slowly.

```python
gamma: float = 0.95
```

This is the discount factor.

It controls how much the agent cares about future rewards.

- A value near `0` means the agent mostly cares about immediate reward.
- A value near `1` means the agent cares a lot about future reward.

Here, `0.95` means future rewards are very important.

```python
epsilon: float = 1.0
```

This controls exploration.

At the start, `epsilon = 1.0`, so the agent mostly chooses random actions.

This helps it explore and learn.

```python
epsilon_min: float = 0.05
```

This is the minimum value of epsilon.

The agent will keep at least `5%` randomness.

```python
epsilon_decay: float = 0.98
```

This controls how fast epsilon decreases.

After each decay, epsilon is multiplied by `0.98`.

So the agent explores less over time.

```python
n_bins: int = 5
```

This controls how many bins are used to simplify continuous state values.

Q-Learning tables work best with discrete states.

But the environment state contains floating-point price changes.

So the code converts those continuous numbers into bins.

---

## Stored Values

```python
self.state_size = state_size
self.n_actions = n_actions
self.alpha = alpha
self.gamma = gamma
self.epsilon = epsilon
self.epsilon_min = epsilon_min
self.epsilon_decay = epsilon_decay
self.n_bins = n_bins
```

These lines store the agent settings.

---

## Q-Table

```python
self.q_table = defaultdict(lambda: np.zeros(n_actions))
```

This creates the Q-table.

It is a dictionary where:

- the key is a state
- the value is an array of Q-values for each action

Example:

```text
(state) -> [0.0, 0.0, 0.0]
```

The array has 3 values because there are 3 actions.

At the beginning, all action values are `0.0`.

---

## Bins

```python
self.bins = np.linspace(-0.3, 0.3, n_bins - 1)
```

This creates bin boundaries between `-0.3` and `0.3`.

These represent normalized price changes from `-30%` to `+30%`.

If `n_bins = 5`, then:

```python
np.linspace(-0.3, 0.3, 4)
```

creates 4 boundary values.

Those boundaries divide price changes into 5 possible groups.

This is used by `_discretize`.

---

## Function: `_discretize`

```python
def _discretize(self, state: np.ndarray) -> tuple:
```

This converts a continuous state into a discrete state.

That means it changes floating-point values into categories.

Q-Learning needs this because a table cannot easily store every possible decimal value.

```python
price_part = state[:-1]
```

This takes all state values except the last one.

The price part contains the normalized price window.

```python
position = int(state[-1])
```

This takes the last state value.

The last value tells whether the agent is holding shares.

It converts it to an integer:

- `0`
- `1`

```python
binned = tuple(np.digitize(s, self.bins) for s in price_part)
```

This converts each price value into a bin number.

`np.digitize` checks where each value belongs compared to the bin boundaries.

Example idea:

- very negative price change -> low bin
- small price change -> middle bin
- very positive price change -> high bin

The result is turned into a tuple so it can be used as a dictionary key.

```python
return binned + (position,)
```

This adds the position value to the binned price values.

The final result is a tuple like:

```text
(2, 2, 3, 4, 1, 0, 2, 3, 3, 4, 1)
```

That tuple represents the simplified state.

---

## Function: `act`

```python
def act(self, state: np.ndarray, greedy: bool = False) -> int:
```

This chooses an action.

It uses epsilon-greedy behavior.

That means:

- sometimes the agent explores with a random action
- otherwise it chooses the best known action

```python
if not greedy and np.random.rand() < self.epsilon:
    return np.random.randint(self.n_actions)
```

If `greedy` is `False` and a random number is less than epsilon, the agent picks a random action.

Example:

If `epsilon = 1.0`, it almost always explores.

If `epsilon = 0.05`, it explores only about 5% of the time.

```python
key = self._discretize(state)
```

If the agent is not exploring, it converts the state into a Q-table key.

```python
return int(np.argmax(self.q_table[key]))
```

This chooses the action with the highest Q-value.

`np.argmax` returns the index of the largest value.

Since the actions are:

- index `0`: hold
- index `1`: buy
- index `2`: sell

The index is the selected action.

If `greedy = True`, the random exploration part is skipped.

So the agent always chooses the best known action.

---

## Function: `update`

```python
def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
```

This updates the Q-table after the agent takes an action.

It uses the Q-Learning formula.

The goal is to make the Q-value for the chosen action closer to the reward plus future expected value.

```python
s = self._discretize(state)
ns = self._discretize(next_state)
```

These convert the current state and next state into table keys.

```python
best_next = float(np.max(self.q_table[ns]))
```

This finds the best Q-value for the next state.

It asks:

```text
From the next state, what is the best value we think we can get?
```

```python
td_target = float(reward) + self.gamma * best_next
```

This calculates the target value.

It combines:

- the immediate reward
- the discounted future reward

This is called the TD target.

TD means temporal difference.

```python
td_error = td_target - float(self.q_table[s][action])
```

This calculates the difference between:

- what the value should move toward
- what the value currently is

This is called the TD error.

```python
self.q_table[s][action] += self.alpha * td_error
```

This updates the Q-value.

The learning rate `alpha` controls how big the update is.

In simple form:

```text
new value = old value + learning rate * error
```

---

## Function: `decay_epsilon`

```python
def decay_epsilon(self):
```

This reduces exploration over time.

```python
self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

It multiplies epsilon by `epsilon_decay`, but never lets it go below `epsilon_min`.

Example:

If:

```text
epsilon = 1.0
epsilon_decay = 0.98
```

After one decay:

```text
epsilon = 0.98
```

After another:

```text
epsilon = 0.9604
```

Over time, the agent becomes less random and more focused on what it has learned.

---

# 3. `dqn.py`

File path:

```text
backend/dqn.py
```

This file defines a Deep Q-Network agent.

A DQN agent is similar to Q-Learning, but instead of storing Q-values in a table, it uses a neural network.

This is useful when the state space is large or continuous.

The environment state contains floating-point price values, so a neural network can handle it more naturally than a Q-table.

---

## Imports

```python
import numpy as np
import random
from collections import deque
```

These imports are used for:

- `numpy`: arrays and numerical work
- `random`: sampling random experiences from memory
- `deque`: storing recent experiences efficiently

---

## Optional PyTorch Import

```python
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
```

The code tries to import PyTorch.

PyTorch is used to build and train the neural network.

If PyTorch is installed:

```python
TORCH_AVAILABLE = True
```

If PyTorch is not installed:

```python
TORCH_AVAILABLE = False
```

This lets the file still load even if PyTorch is missing.

However, the real DQN training only works when PyTorch is available.

---

## Class: `DQNNetwork`

```python
class DQNNetwork(nn.Module if TORCH_AVAILABLE else object):
```

This class defines the neural network used by the DQN agent.

If PyTorch is available, it inherits from:

```python
nn.Module
```

That is the base class for PyTorch neural networks.

If PyTorch is not available, it inherits from:

```python
object
```

But in that case, creating the network will raise an error.

---

## Constructor: `DQNNetwork.__init__`

```python
def __init__(self, input_size: int, output_size: int):
```

This creates the neural network.

### Parameters

```python
input_size
```

The number of values in the state.

For example, if the environment uses:

```text
window_size = 10
```

Then:

```text
input_size = 11
```

because the state has 10 price values plus 1 position value.

```python
output_size
```

The number of actions.

Usually:

```text
output_size = 3
```

because the actions are hold, buy, and sell.

```python
if not TORCH_AVAILABLE:
    raise RuntimeError("PyTorch not available")
```

If PyTorch is not installed, the network cannot be created.

So the code raises an error.

```python
super().__init__()
```

This initializes the PyTorch parent class.

---

## Network Layers

```python
self.net = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, output_size),
)
```

This builds a simple feed-forward neural network.

It has:

1. An input layer
2. A hidden layer with 128 neurons
3. A ReLU activation
4. A hidden layer with 64 neurons
5. Another ReLU activation
6. An output layer

### Layer 1

```python
nn.Linear(input_size, 128)
```

This takes the state and transforms it into 128 values.

### Activation 1

```python
nn.ReLU()
```

ReLU is an activation function.

It helps the network learn non-linear patterns.

In simple terms, it allows the network to learn more complex relationships than a straight line.

### Layer 2

```python
nn.Linear(128, 64)
```

This transforms the 128 values into 64 values.

### Activation 2

```python
nn.ReLU()
```

Another ReLU activation.

### Output Layer

```python
nn.Linear(64, output_size)
```

This produces one output value for each action.

If there are 3 actions, the network outputs 3 Q-values:

```text
[Q(hold), Q(buy), Q(sell)]
```

The action with the highest Q-value is considered the best action.

---

## Function: `forward`

```python
def forward(self, x):
    return self.net(x)
```

This defines how data passes through the network.

The input `x` goes through all the layers in `self.net`.

The output is the predicted Q-values for the actions.

---

## Class: `DQNAgent`

```python
class DQNAgent:
```

This class controls the DQN learning process.

It handles:

- choosing actions
- storing experiences
- training from past experiences
- updating the target network
- reducing exploration over time

---

## Constructor: `DQNAgent.__init__`

```python
def __init__(self, state_size: int, n_actions: int = 3, gamma: float = 0.95,
             epsilon: float = 1.0, epsilon_min: float = 0.05,
             epsilon_decay: float = 0.98, lr: float = 5e-4,
             batch_size: int = 64, memory_size: int = 5000,
             target_update_freq: int = 10, replay_every: int = 4):
```

This creates the DQN agent and sets all training settings.

### Parameters

```python
state_size
```

The number of values in the environment state.

Usually:

```text
window_size + 1
```

```python
n_actions = 3
```

The number of actions:

- hold
- buy
- sell

```python
gamma = 0.95
```

The discount factor.

It controls how much future rewards matter.

```python
epsilon = 1.0
```

The starting exploration rate.

At the beginning, the agent often chooses random actions.

```python
epsilon_min = 0.05
```

The minimum exploration rate.

The agent keeps at least 5% randomness.

```python
epsilon_decay = 0.98
```

How fast epsilon decreases.

```python
lr = 5e-4
```

This is the learning rate for the neural network optimizer.

`5e-4` means:

```text
0.0005
```

```python
batch_size = 64
```

The agent trains using 64 experiences at a time.

```python
memory_size = 5000
```

The replay memory stores up to 5000 experiences.

Older experiences are removed when the memory is full.

```python
target_update_freq = 10
```

This controls how often the target network is updated.

```python
replay_every = 4
```

This means training happens every 4 stored steps, not every single step.

This can make training faster.

---

## Stored Values

```python
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
```

These lines save the settings inside the agent.

```python
self.memory = deque(maxlen=memory_size)
```

This creates replay memory.

Replay memory stores experiences in this format:

```text
(state, action, reward, next_state, done)
```

The agent later samples random experiences from this memory to train.

This is called experience replay.

Experience replay helps because the agent can learn from old experiences multiple times.

---

## If PyTorch Is Available

```python
if TORCH_AVAILABLE:
```

If PyTorch is installed, the agent creates neural networks and training tools.

```python
self.device = torch.device("cpu")
```

This tells PyTorch to use the CPU.

The code does not use GPU here.

```python
self.policy_net = DQNNetwork(state_size, n_actions).to(self.device)
```

This creates the policy network.

The policy network is the main network that gets trained.

It predicts Q-values and chooses actions.

```python
self.target_net = DQNNetwork(state_size, n_actions).to(self.device)
```

This creates the target network.

The target network is a copy of the policy network.

It is used to make training more stable.

```python
self.target_net.load_state_dict(self.policy_net.state_dict())
```

This copies the policy network's weights into the target network.

At the start, both networks are the same.

```python
self.target_net.eval()
```

This puts the target network in evaluation mode.

That means it is not being trained directly.

```python
self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
```

This creates the Adam optimizer.

The optimizer updates the policy network weights during training.

```python
self.loss_fn = nn.SmoothL1Loss()
```

This creates the loss function.

`SmoothL1Loss` is also called Huber loss.

It is often used in DQN because it is less sensitive to very large errors than normal squared loss.

---

## If PyTorch Is Not Available

```python
else:
    self.q_table_fallback = {}
```

If PyTorch is not installed, the agent creates an empty fallback dictionary.

However, this DQN agent does not really implement full table-based fallback learning.

In this case:

- `act` just returns random actions when no neural network is available
- `replay` does nothing

So real DQN learning requires PyTorch.

---

## Function: `remember`

```python
def remember(self, state, action, reward, next_state, done):
```

This stores one experience in replay memory.

```python
self.memory.append((state, action, float(reward), next_state, done))
```

The stored experience contains:

- current state
- action taken
- reward received
- next state
- whether the episode ended

The reward is converted to `float`.

```python
self.steps += 1
```

The step counter increases by 1.

This counter is used to decide when replay training should happen.

---

## Function: `act`

```python
def act(self, state: np.ndarray, greedy: bool = False) -> int:
```

This chooses an action.

Like the Q-Learning agent, it uses epsilon-greedy behavior.

```python
if not greedy and np.random.rand() < self.epsilon:
    return np.random.randint(self.n_actions)
```

If the agent is exploring, it picks a random action.

This helps the agent discover what works.

```python
if TORCH_AVAILABLE:
```

If PyTorch is available, the agent uses the neural network.

```python
with torch.no_grad():
```

This means PyTorch should not track gradients.

Gradients are needed for training, but not for simply choosing an action.

Using `torch.no_grad()` makes action selection faster and uses less memory.

```python
t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
```

This converts the NumPy state into a PyTorch tensor.

`unsqueeze(0)` adds a batch dimension.

The network expects a batch of states, even if there is only one state.

Example:

```text
[11 values]
```

becomes:

```text
[[11 values]]
```

```python
q_vals = self.policy_net(t)
```

The policy network predicts Q-values for the actions.

Example:

```text
[0.2, 1.5, -0.3]
```

This means:

- hold has value `0.2`
- buy has value `1.5`
- sell has value `-0.3`

```python
return int(q_vals.argmax().item())
```

This returns the action with the highest Q-value.

In the example above, the agent chooses action `1`, which means buy.

```python
else:
    return np.random.randint(self.n_actions)
```

If PyTorch is not available, the agent returns a random action.

---

## Function: `replay`

```python
def replay(self):
```

This trains the DQN agent using experiences stored in memory.

This is the main learning function for the neural network.

---

## Replay Timing

```python
if self.steps % self.replay_every != 0:
    return
```

The agent only trains every few steps.

If `replay_every = 4`, then training happens every 4 steps.

This saves time.

```python
if len(self.memory) < self.batch_size:
    return
```

The agent does not train until it has enough experiences.

If `batch_size = 64`, it waits until memory has at least 64 experiences.

```python
if not TORCH_AVAILABLE:
    return
```

If PyTorch is not installed, training cannot happen.

So the function stops.

---

## Sampling a Batch

```python
batch = random.sample(self.memory, self.batch_size)
```

This randomly selects experiences from replay memory.

Random sampling helps break the strong order of time-based data.

That makes training more stable.

```python
states, actions, rewards, next_states, dones = zip(*batch)
```

This separates the batch into five groups:

- states
- actions
- rewards
- next states
- done flags

---

## Converting Batch Data to Tensors

```python
states_t = torch.FloatTensor(np.array(states)).to(self.device)
actions_t = torch.LongTensor(actions).to(self.device)
rewards_t = torch.FloatTensor(rewards).to(self.device)
next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
dones_t = torch.BoolTensor(dones).to(self.device)
```

These lines convert the batch data into PyTorch tensors.

PyTorch needs tensors to train neural networks.

`actions_t` uses `LongTensor` because action indexes must be integer values.

`dones_t` uses `BoolTensor` because `done` is true or false.

---

## Current Q-Values

```python
current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
```

This calculates the Q-values predicted by the policy network for the actions that were actually taken.

Step by step:

```python
self.policy_net(states_t)
```

This predicts Q-values for all actions for every state in the batch.

Example:

```text
[
  [0.1, 0.5, -0.2],
  [0.3, -0.1, 0.7]
]
```

```python
actions_t.unsqueeze(1)
```

This reshapes the actions so `gather` can use them.

```python
.gather(1, actions_t.unsqueeze(1))
```

This selects only the Q-value for the action that was actually taken.

```python
.squeeze(1)
```

This removes an unnecessary dimension.

The final result is one current Q-value per experience.

---

## Target Q-Values

```python
with torch.no_grad():
    next_q = self.target_net(next_states_t).max(1)[0]
    next_q[dones_t] = 0.0
    target_q = rewards_t + self.gamma * next_q
```

This calculates what the Q-values should move toward.

```python
next_q = self.target_net(next_states_t).max(1)[0]
```

The target network predicts Q-values for the next states.

`.max(1)[0]` selects the best Q-value for each next state.

```python
next_q[dones_t] = 0.0
```

If an experience ended the episode, there is no future reward.

So its future Q-value is set to `0.0`.

```python
target_q = rewards_t + self.gamma * next_q
```

This is the DQN target formula:

```text
target = reward + gamma * best future Q-value
```

This is similar to the Q-Learning update, but done with neural network predictions.

---

## Loss Calculation

```python
loss = self.loss_fn(current_q, target_q)
```

This compares:

- the current predicted Q-values
- the target Q-values

The loss measures how wrong the network is.

The optimizer will try to reduce this loss.

---

## Training the Network

```python
self.optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
self.optimizer.step()
```

These lines train the policy network.

```python
self.optimizer.zero_grad()
```

This clears old gradients.

Gradients from the previous training step should not be reused.

```python
loss.backward()
```

This calculates gradients.

Gradients tell the optimizer how to change the network weights to reduce the loss.

```python
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
```

This clips the gradients so they do not become too large.

Large gradients can make training unstable.

```python
self.optimizer.step()
```

This updates the policy network weights.

---

## Updating the Target Network

```python
if (self.steps // self.replay_every) % self.target_update_freq == 0:
    self.target_net.load_state_dict(self.policy_net.state_dict())
```

This occasionally copies the policy network weights into the target network.

The target network is not updated every training step.

This helps stabilize learning.

If the target network changed constantly, the learning target would move too much.

---

## Function: `decay_epsilon`

```python
def decay_epsilon(self):
```

This reduces exploration over time.

```python
self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

It multiplies epsilon by `epsilon_decay`, but never lets epsilon go below `epsilon_min`.

So the agent starts with more random actions, then slowly becomes more confident and uses what it learned.

---

# How the Three Files Work Together

The full training loop usually works like this:

1. Create the trading environment using prices.
2. Reset the environment to get the first state.
3. Give the state to an agent.
4. The agent chooses an action.
5. The environment applies the action using `step`.
6. The environment returns the next state, reward, and done flag.
7. The agent learns from that experience.
8. Repeat until the episode is done.

For Q-Learning:

```python
agent.update(state, action, reward, next_state)
```

updates the Q-table directly.

For DQN:

```python
agent.remember(state, action, reward, next_state, done)
agent.replay()
```

stores the experience and trains the neural network from replay memory.

---

# Main Difference Between Q-Learning and DQN

## Q-Learning Agent

The Q-Learning agent uses a table.

Advantages:

- simple
- easy to understand
- good for small state spaces

Disadvantages:

- struggles with many continuous values
- needs discretization
- Q-table can become very large

## DQN Agent

The DQN agent uses a neural network.

Advantages:

- better for continuous state values
- can generalize between similar states
- more powerful than a basic table

Disadvantages:

- needs PyTorch
- more complex
- needs more training care

---

# Important Variables Summary

## Environment Variables

```text
prices
```

The price data used for trading.

```text
window_size
```

How many previous prices the agent sees.

```text
balance
```

The cash money available.

```text
shares_held
```

How many shares the agent owns.

```text
buy_price
```

The price where the current shares were bought.

```text
prev_net_worth
```

The previous total portfolio value.

---

## Agent Variables

```text
epsilon
```

Controls random exploration.

```text
epsilon_min
```

The smallest allowed exploration rate.

```text
epsilon_decay
```

How quickly exploration decreases.

```text
gamma
```

How much future rewards matter.

```text
alpha
```

Learning rate for Q-Learning.

```text
lr
```

Learning rate for the DQN neural network optimizer.

```text
batch_size
```

How many experiences DQN trains on at once.

```text
memory
```

The replay memory used by DQN.

---

# Simple Example of One Step

Imagine:

```text
current price = 100
balance = 10000
shares_held = 0
action = 1
```

Action `1` means buy.

The environment calculates:

```text
10000 // 100 = 100 shares
```

So the agent buys 100 shares.

Now:

```text
balance = 0
shares_held = 100
buy_price = 100
```

If later the price becomes `110` and the agent sells:

```text
revenue = 100 * 110 = 11000
profit percentage = (110 - 100) / 100 = 0.10
realized reward = 0.10 * 10 = 1.0
```

The agent receives a positive reward because it made a profit.

If the price became `90` instead:

```text
profit percentage = (90 - 100) / 100 = -0.10
realized reward = -0.10 * 10 = -1.0
```

The agent receives a negative reward because it lost money.

---

# Final Simple Summary

`env.py` is the trading world.

It keeps track of prices, balance, shares, portfolio value, and rewards.

`q_learning.py` is a simple table-based learning agent.

It converts states into bins and learns Q-values in a dictionary.

`dqn.py` is a neural-network-based learning agent.

It stores experiences, samples batches, trains a policy network, and uses a target network for stable learning.

Together, these files create a basic reinforcement learning system where an agent learns trading actions from reward feedback.
