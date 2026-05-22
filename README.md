# RL Trading System

A full-stack reinforcement learning trading project that trains an agent to make simple market decisions: hold, buy, or sell.

The project includes:

- A Python FastAPI backend
- A React + Vite frontend
- A custom trading environment
- Q-Learning and Deep Q-Network agents
- Market data loading from Yahoo Finance and CoinGecko
- A dashboard for training, simulation, charts, and portfolio statistics

---

## Team Members

| Name | ID |
|---|---|
| Ziad Hany Saeed Othman | 20230120 |
| Ahmed Yasser Hassanein El-Azouk | 20230070 |
| Omar Mohamed Qamar Eldawla Gomaa Gadallah | 20230218 |

---

## Project Idea

The goal of this project is to show how reinforcement learning can be used in a simple trading system.

The agent receives a state that represents recent price movement and whether it currently owns shares. Then it chooses one of three actions:

| Action | Meaning |
|---:|---|
| 0 | Hold |
| 1 | Buy |
| 2 | Sell |

The environment gives the agent a reward based on portfolio performance, realized profit or loss, and invalid actions.

---

## Technologies Used

### Backend

- Python
- FastAPI
- Uvicorn
- NumPy
- PyTorch
- yfinance
- pandas
- Pydantic

### Frontend

- React
- TypeScript
- Vite
- Tailwind CSS
- Recharts

---

## Project Structure

```text
rl-trading/
├── backend/
│   ├── main.py
│   ├── env.py
│   ├── q_learning.py
│   ├── dqn.py
│   ├── data_loader.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── index.css
│   │   └── pages/
│   │       ├── Hero.tsx
│   │       ├── Dashboard.tsx
│   │       ├── Chart.tsx
│   │       └── Stats.tsx
│   ├── package.json
│   ├── vite.config.js
│   └── tsconfig.json
│
├── README.txt
├── README.md
└── file.md
```

---

## Backend Explanation

### `main.py`

This file runs the FastAPI server.

It provides API endpoints for:

- starting training
- checking training status
- running simulation
- checking backend health

It also stores trained agents in memory.

### `env.py`

This file contains `TradingEnv`.

The environment handles:

- price window state
- current balance
- shares held
- buy and sell actions
- reward calculation
- portfolio value
- episode ending

### `q_learning.py`

This file contains `QLearningAgent`.

It uses a Q-table to learn action values. Since price states are continuous numbers, the agent first converts them into bins.

### `dqn.py`

This file contains:

- `DQNNetwork`
- `DQNAgent`

The DQN agent uses a neural network instead of a Q-table. It also uses replay memory and a target network to make learning more stable.

### `data_loader.py`

This file loads price data.

It supports:

- stocks from Yahoo Finance
- crypto prices from CoinGecko
- fallback demo data if live data fails

Supported tickers:

```text
AAPL, MSFT, GOOGL, BTC-USD, ETH-USD, TSLA, SPY
```

---

## Frontend Explanation

### `App.tsx`

Controls whether the user sees the landing page or the dashboard.

### `Hero.tsx`

Displays the landing screen with the project title and an animated background.

### `Dashboard.tsx`

The main user interface.

It lets the user:

- select a ticker
- select an algorithm
- choose number of episodes
- choose the training period
- train the agent
- run simulation
- view progress and errors

### `Chart.tsx`

Displays the price and portfolio chart using Recharts.

It also shows buy and sell signals.

### `Stats.tsx`

Displays the final simulation statistics:

- final portfolio value
- total profit or loss
- return percentage
- total trades

---

## API Endpoints

Base URL:

```text
http://localhost:8000/rl-api
```

### Train Agent

```http
POST /train
```

Request body:

```json
{
  "ticker": "AAPL",
  "algorithm": "q_learning",
  "episodes": 30,
  "period": "3mo"
}
```

Returns a task ID used to check training progress.

### Check Training Status

```http
GET /train/status/{task_id}
```

Returns:

- status
- progress
- current episode
- total episodes
- last reward
- number of data points

### Run Simulation

```http
POST /simulate
```

Request body:

```json
{
  "ticker": "AAPL",
  "algorithm": "q_learning"
}
```

Returns:

- prices
- portfolio values
- buy signals
- sell signals
- final statistics

### Health Check

```http
GET /healthz
```

Returns:

```json
{
  "status": "ok"
}
```

---

## How to Run the Project

### 1. Start the Backend

Open a terminal:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend should run on:

```text
http://localhost:8000
```

### 2. Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend should run on:

```text
http://localhost:3000
```

---

## How to Use

1. Open the frontend in the browser.
2. Click `ENTER DASHBOARD`.
3. Choose a ticker.
4. Choose an algorithm: `Q-Learning` or `DQN`.
5. Choose the number of episodes.
6. Choose the period.
7. Click `TRAIN`.
8. Wait until training is complete.
9. Click `RUN SIMULATION`.
10. View the chart, portfolio value, buy/sell signals, and statistics.

---

## Algorithms

### Q-Learning

Q-Learning is simple and table-based.

It stores values for each state-action pair in a Q-table.

Advantages:

- simple
- fast
- easy to understand

Limitations:

- needs discretization
- does not handle large continuous state spaces very well

### Deep Q-Network

DQN uses a neural network to predict Q-values.

Advantages:

- handles continuous states better
- can generalize between similar states
- more powerful than a simple table

Limitations:

- slower to train
- needs PyTorch
- more complex

---

## Notes

- Q-Learning is usually faster for quick tests.
- DQN may take more time, especially with longer periods or many episodes.
- The project is educational and simplified.
- It does not include real trading fees, slippage, risk control, or live trading execution.

---

## Future Improvements

Possible improvements:

- save trained models to disk
- add transaction fees
- add benchmark comparison
- add Sharpe ratio and drawdown
- add technical indicators
- add more tickers
- add model performance history
- add user authentication
- deploy backend and frontend online
