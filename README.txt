# RL Trading System

Reinforcement Learning Trading System with Q-Learning and DQN algorithms.

==============================================================
 REQUIREMENTS
==============================================================

Backend:
  - Python 3.10 or newer
  - pip

Frontend:
  - Node.js 18 or newer
  - npm

==============================================================
 STEP 1 — Set up the Backend
==============================================================

Open a terminal and run:

  cd rl-trading/backend

  pip install -r requirements.txt

  NOTE: PyTorch installation can take a few minutes.
        If you only want CPU support (no GPU), run instead:
        pip install torch --index-url https://download.pytorch.org/whl/cpu

  Once installed, start the server:

  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  You should see:
    INFO: Uvicorn running on http://0.0.0.0:8000

  Leave this terminal open.

==============================================================
 STEP 2 — Set up the Frontend
==============================================================

Open a SECOND terminal and run:

  cd rl-trading/frontend

  npm install

  npm run dev

  You should see:
    VITE ready in ...ms
    Local: http://localhost:3000/

  Open http://localhost:3000 in your browser.

==============================================================
 HOW TO USE
==============================================================

1. Click "ENTER DASHBOARD"

2. In the "Train Agent" section:
   - Pick a ticker  (AAPL, MSFT, BTC-USD, etc.)
   - Pick algorithm (Q-Learning or DQN)
   - Set episodes   (20-50 is a good start)
   - Pick period    (3 Months is fastest; 1 Year is more data)
   - Click TRAIN
   - Watch the progress bar fill up

3. Once training says "Training complete":
   - Click "RUN SIMULATION"
   - See the price chart, portfolio line, buy/sell signals
   - Check the stats: final value, profit, number of trades

==============================================================
 SPEED TIPS
==============================================================

- Q-Learning trains much faster than DQN
- Use "1 Month" or "3 Months" period for quick results
- DQN with "2 Years" and 50+ episodes can take several minutes
- Start with 20-30 episodes; increase for better learning

==============================================================
 PROJECT STRUCTURE
==============================================================

rl-trading/
├── backend/
│   ├── main.py          FastAPI server + API endpoints
│   ├── env.py           Trading environment (Gym-style)
│   ├── q_learning.py    Q-Learning agent (Q-table)
│   ├── dqn.py           DQN agent (PyTorch neural network)
│   ├── data_loader.py   yfinance price data loader
│   └── requirements.txt Python dependencies
│
└── frontend/
    ├── src/
    │   ├── App.tsx           Root component
    │   ├── index.css         Dark neon theme (Tailwind v4)
    │   └── pages/
    │       ├── Hero.tsx      Landing screen
    │       ├── Dashboard.tsx Train + simulate controls
    │       ├── Chart.tsx     Recharts price/portfolio chart
    │       └── Stats.tsx     Portfolio stats cards
    ├── index.html
    ├── vite.config.js
    ├── tsconfig.json
    └── package.json

==============================================================
 API ENDPOINTS (http://localhost:8000)
==============================================================

POST /train
  Body: { ticker, algorithm, episodes, period }
  Returns: { task_id, status }

GET  /train/status/{task_id}
  Returns: { status, progress, episode, episodes, last_reward }

POST /simulate
  Body: { ticker, algorithm }
  Returns: { prices, portfolio_values, buy_signals, sell_signals, stats }

GET  /healthz
  Returns: { status: "ok" }

