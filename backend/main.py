from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from data_loader import load_data
from env import TradingEnv
from q_learning import QLearningAgent
from dqn import DQNAgent

app = FastAPI(title="RL Trading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_status = {}
trained_agents = {}
trained_meta = {}


class TrainRequest(BaseModel):
    ticker: str
    algorithm: str
    episodes: int = 50
    period: str = "3mo"


class SimulateRequest(BaseModel):
    ticker: str
    algorithm: str


def run_training(task_id: str, ticker: str, algorithm: str, episodes: int, period: str):
    training_status[task_id] = {"status": "running", "progress": 0, "episode": 0, "episodes": episodes}
    try:
        prices = load_data(ticker, period=period)
        env = TradingEnv(prices)
        state_size = env.window_size + 1  # window + position indicator

        if algorithm == "q_learning":
            agent = QLearningAgent(state_size=state_size)
        else:
            agent = DQNAgent(state_size=state_size)

        rewards = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                if algorithm == "dqn":
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay()
                else:
                    agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += float(reward)
            rewards.append(round(total_reward, 4))
            agent.decay_epsilon()
            progress = int(((ep + 1) / episodes) * 100)
            training_status[task_id] = {
                "status": "running",
                "progress": progress,
                "episode": ep + 1,
                "episodes": episodes,
                "last_reward": round(total_reward, 4),
                "data_points": int(len(prices)),
            }

        key = f"{ticker}_{algorithm}"
        trained_agents[key] = agent
        trained_meta[key] = {"period": period}
        training_status[task_id] = {
            "status": "done",
            "progress": 100,
            "episode": episodes,
            "episodes": episodes,
            "rewards": rewards,
            "data_points": int(len(prices)),
        }
    except Exception as e:
        training_status[task_id] = {"status": "error", "message": str(e)}


@app.post("/rl-api/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    task_id = f"{req.ticker}_{req.algorithm}_{req.episodes}_{req.period}"
    background_tasks.add_task(
        run_training, task_id, req.ticker, req.algorithm, req.episodes, req.period
    )
    return {"task_id": task_id, "status": "started"}


@app.get("/rl-api/train/status/{task_id}")
def get_status(task_id: str):
    if task_id not in training_status:
        raise HTTPException(status_code=404, detail="Task not found")
    return training_status[task_id]


@app.post("/rl-api/simulate")
def simulate(req: SimulateRequest):
    key = f"{req.ticker}_{req.algorithm}"
    agent = trained_agents.get(key)
    if agent is None:
        raise HTTPException(status_code=400, detail="Agent not trained yet. Train first.")

    meta = trained_meta.get(key, {})
    period = meta.get("period", "3mo")

    prices = load_data(req.ticker, period=period)
    env = TradingEnv(prices)
    state = env.reset()
    done = False

    portfolio_values: list[float] = [float(env.initial_balance)]
    buy_signals: list[int] = []
    sell_signals: list[int] = []

    while not done:
        action = agent.act(state, greedy=True)
        next_state, reward, done = env.step(action)
        portfolio_values.append(float(env.portfolio_value()))
        if action == 1:
            buy_signals.append(len(portfolio_values) - 1)
        elif action == 2:
            sell_signals.append(len(portfolio_values) - 1)
        state = next_state

    price_list: list[float] = [float(p) for p in prices[env.window_size:]]
    final_value = portfolio_values[-1]
    initial_value = portfolio_values[0]
    profit = final_value - initial_value
    num_trades = len(buy_signals) + len(sell_signals)

    return {
        "prices": price_list,
        "portfolio_values": portfolio_values,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "stats": {
            "final_portfolio_value": round(float(final_value), 2),
            "total_profit": round(float(profit), 2),
            "total_profit_pct": round(float(profit / initial_value) * 100, 2),
            "num_trades": num_trades,
            "initial_balance": float(initial_value),
        },
    }


@app.get("/rl-api/healthz")
def health():
    return {"status": "ok"}
