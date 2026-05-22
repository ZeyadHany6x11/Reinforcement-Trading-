import { useState, useRef, useCallback } from "react";
import Chart from "./Chart";
import Stats from "./Stats";

const TICKERS = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD", "TSLA", "SPY"];
const ALGORITHMS = [
  { value: "q_learning", label: "Q-Learning" },
  { value: "dqn", label: "Deep Q-Network (DQN)" },
];
const PERIODS = [
  { value: "1mo", label: "1 Month", note: "~22 days · fastest" },
  { value: "3mo", label: "3 Months", note: "~66 days · fast" },
  { value: "6mo", label: "6 Months", note: "~130 days · medium" },
  { value: "1y",  label: "1 Year",   note: "~252 days · slower" },
  { value: "2y",  label: "2 Years",  note: "~504 days · slow" },
];

const API_BASE = "http://localhost:8000/rl-api";

interface SimResult {
  prices: number[];
  portfolio_values: number[];
  buy_signals: number[];
  sell_signals: number[];
  stats: {
    final_portfolio_value: number;
    total_profit: number;
    total_profit_pct: number;
    num_trades: number;
    initial_balance: number;
  };
}

interface TrainingStatus {
  status: string;
  progress: number;
  episode: number;
  episodes: number;
  last_reward?: number;
  message?: string;
  rewards?: number[];
  data_points?: number;
}

export default function Dashboard() {
  const [ticker, setTicker] = useState("AAPL");
  const [algorithm, setAlgorithm] = useState("q_learning");
  const [episodes, setEpisodes] = useState(30);
  const [period, setPeriod] = useState("3mo");
  const [training, setTraining] = useState(false);
  const [trainStatus, setTrainStatus] = useState<TrainingStatus | null>(null);
  const [trained, setTrained] = useState(false);
  const [simResult, setSimResult] = useState<SimResult | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPoll = () => {
    if (pollRef.current) clearInterval(pollRef.current);
  };

  const handleTrain = useCallback(async () => {
    setTraining(true);
    setTrainError(null);
    setTrainStatus(null);
    setTrained(false);
    setSimResult(null);

    try {
      const res = await fetch(`${API_BASE}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, algorithm, episodes, period }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Training request failed");
      }
      const { task_id } = await res.json();

      pollRef.current = setInterval(async () => {
        try {
          const sr = await fetch(`${API_BASE}/train/status/${task_id}`);
          if (!sr.ok) return;
          const status: TrainingStatus = await sr.json();
          setTrainStatus(status);
          if (status.status === "done") {
            stopPoll();
            setTraining(false);
            setTrained(true);
          } else if (status.status === "error") {
            stopPoll();
            setTraining(false);
            setTrainError(status.message || "Unknown error");
          }
        } catch {}
      }, 800);
    } catch (e: any) {
      setTrainError(e.message);
      setTraining(false);
    }
  }, [ticker, algorithm, episodes, period]);

  const handleSimulate = useCallback(async () => {
    setSimLoading(true);
    setSimError(null);
    try {
      const res = await fetch(`${API_BASE}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, algorithm }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Simulation failed");
      }
      const data: SimResult = await res.json();
      setSimResult(data);
    } catch (e: any) {
      setSimError(e.message);
    } finally {
      setSimLoading(false);
    }
  }, [ticker, algorithm]);

  const algoLabel = ALGORITHMS.find((a) => a.value === algorithm)?.label ?? algorithm;
  const periodLabel = PERIODS.find((p) => p.value === period)?.label ?? period;

  return (
    <div className="min-h-screen" style={{ background: "#0d1117" }}>
      {/* Top bar */}
      <div
        className="sticky top-0 z-20 flex items-center justify-between px-6 py-3"
        style={{
          background: "rgba(13,17,23,0.92)",
          borderBottom: "1px solid rgba(0,255,255,0.1)",
          backdropFilter: "blur(12px)",
        }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-2 h-2 rounded-full"
            style={{ background: "#00ffff", boxShadow: "0 0 8px #00ffff" }}
          />
          <span
            className="font-black text-sm tracking-widest uppercase font-mono"
            style={{ color: "#00ffff", textShadow: "0 0 10px rgba(0,255,255,0.5)" }}
          >
            RL TRADING SYSTEM
          </span>
        </div>
        <div className="flex gap-3">
          {["Q-LEARNING", "DQN"].map((a) => (
            <span
              key={a}
              className="text-xs font-mono px-2 py-1 rounded"
              style={{
                background: "rgba(0,255,255,0.06)",
                border: "1px solid rgba(0,255,255,0.12)",
                color: "rgba(0,255,255,0.45)",
              }}
            >
              {a}
            </span>
          ))}
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
        {/* Train Agent */}
        <section
          className="glass-card rounded-lg p-6"
          style={{
            background: "rgba(13,17,23,0.85)",
            border: "1px solid rgba(0,255,255,0.12)",
          }}
        >
          <div className="flex items-center gap-2 mb-5">
            <div className="w-1 h-5 rounded" style={{ background: "#00ffff", boxShadow: "0 0 6px #00ffff" }} />
            <h2
              className="text-sm font-bold tracking-widest uppercase font-mono"
              style={{ color: "#00ffff" }}
            >
              Train Agent
            </h2>
          </div>

          {/* Controls grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            <div>
              <label className="block text-xs font-mono mb-2 tracking-widest uppercase" style={{ color: "rgba(255,255,255,0.35)" }}>
                Ticker
              </label>
              <select
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                disabled={training}
                className="neon-select w-full rounded px-3 py-2 text-sm font-mono"
              >
                {TICKERS.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-mono mb-2 tracking-widest uppercase" style={{ color: "rgba(255,255,255,0.35)" }}>
                Algorithm
              </label>
              <select
                value={algorithm}
                onChange={(e) => setAlgorithm(e.target.value)}
                disabled={training}
                className="neon-select w-full rounded px-3 py-2 text-sm font-mono"
              >
                {ALGORITHMS.map((a) => (
                  <option key={a.value} value={a.value}>{a.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-mono mb-2 tracking-widest uppercase" style={{ color: "rgba(255,255,255,0.35)" }}>
                Episodes
              </label>
              <input
                type="number"
                value={episodes}
                onChange={(e) => setEpisodes(Number(e.target.value))}
                min={5}
                max={500}
                disabled={training}
                className="neon-input w-full rounded px-3 py-2 text-sm font-mono"
              />
            </div>
            <div>
              <label className="block text-xs font-mono mb-2 tracking-widest uppercase" style={{ color: "rgba(255,255,255,0.35)" }}>
                Training Period
              </label>
              <select
                value={period}
                onChange={(e) => setPeriod(e.target.value)}
                disabled={training}
                className="neon-select w-full rounded px-3 py-2 text-sm font-mono"
              >
                {PERIODS.map((p) => (
                  <option key={p.value} value={p.value}>{p.label}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Period hint */}
          <div className="mb-5">
            {(() => {
              const p = PERIODS.find((x) => x.value === period);
              return p ? (
                <div
                  className="inline-flex items-center gap-2 px-3 py-1.5 rounded text-xs font-mono"
                  style={{
                    background: "rgba(0,255,255,0.04)",
                    border: "1px solid rgba(0,255,255,0.12)",
                    color: "rgba(0,255,255,0.5)",
                  }}
                >
                  <span style={{ color: "rgba(0,255,255,0.3)" }}>Period:</span>
                  <span style={{ color: "#00ffff" }}>{p.label}</span>
                  <span style={{ color: "rgba(0,255,255,0.35)" }}>—</span>
                  <span>{p.note}</span>
                </div>
              ) : null;
            })()}
          </div>

          <button
            onClick={handleTrain}
            disabled={training}
            className="neon-btn-primary px-8 py-2.5 rounded text-sm font-bold tracking-widest uppercase font-mono"
          >
            {training ? "TRAINING..." : "TRAIN"}
          </button>

          {training && trainStatus && (
            <div className="mt-5 space-y-2">
              <div className="flex justify-between text-xs font-mono" style={{ color: "rgba(255,255,255,0.45)" }}>
                <span>
                  Episode {trainStatus.episode} / {trainStatus.episodes}
                  {trainStatus.data_points && (
                    <span style={{ color: "rgba(0,255,255,0.35)", marginLeft: 8 }}>
                      · {trainStatus.data_points} data pts
                    </span>
                  )}
                </span>
                <span>{trainStatus.progress}%</span>
              </div>
              <div
                className="w-full h-1.5 rounded-full overflow-hidden"
                style={{ background: "rgba(0,255,255,0.1)" }}
              >
                <div
                  className="progress-bar-inner h-full rounded-full"
                  style={{ width: `${trainStatus.progress}%` }}
                />
              </div>
              {trainStatus.last_reward !== undefined && (
                <div className="text-xs font-mono" style={{ color: "rgba(0,255,255,0.5)" }}>
                  Last reward: {trainStatus.last_reward.toFixed(2)}
                </div>
              )}
            </div>
          )}

          {trainStatus?.status === "done" && (
            <div
              className="mt-4 px-4 py-2 rounded text-xs font-mono"
              style={{
                background: "rgba(0,255,136,0.06)",
                border: "1px solid rgba(0,255,136,0.2)",
                color: "#00ff88",
              }}
            >
              Training complete — {trainStatus.episodes} episodes · {ticker} · {algoLabel} · {periodLabel}
              {trainStatus.data_points && (
                <span style={{ color: "rgba(0,255,136,0.55)" }}> · {trainStatus.data_points} data points</span>
              )}
            </div>
          )}

          {trainError && (
            <div
              className="mt-4 px-4 py-2 rounded text-xs font-mono"
              style={{
                background: "rgba(255,68,102,0.06)",
                border: "1px solid rgba(255,68,102,0.2)",
                color: "#ff4466",
              }}
            >
              Error: {trainError}
            </div>
          )}
        </section>

        {/* Simulation */}
        <section
          className="glass-card rounded-lg p-6"
          style={{
            background: "rgba(13,17,23,0.85)",
            border: "1px solid rgba(0,255,255,0.12)",
          }}
        >
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2">
              <div className="w-1 h-5 rounded" style={{ background: "#a78bfa", boxShadow: "0 0 6px #a78bfa" }} />
              <h2
                className="text-sm font-bold tracking-widest uppercase font-mono"
                style={{ color: "#a78bfa" }}
              >
                Simulation
              </h2>
            </div>
            <button
              onClick={handleSimulate}
              disabled={!trained || simLoading}
              className="px-6 py-2 rounded text-sm font-bold tracking-widest uppercase font-mono transition-all duration-200"
              style={{
                background: trained ? "rgba(167,139,250,0.1)" : "rgba(255,255,255,0.03)",
                border: `1px solid ${trained ? "#a78bfa" : "rgba(255,255,255,0.1)"}`,
                color: trained ? "#a78bfa" : "rgba(255,255,255,0.2)",
                boxShadow: trained ? "0 0 12px rgba(167,139,250,0.2)" : "none",
                cursor: trained ? "pointer" : "not-allowed",
              }}
            >
              {simLoading ? "RUNNING..." : "RUN SIMULATION"}
            </button>
          </div>

          {!trained && (
            <div className="text-xs font-mono" style={{ color: "rgba(255,255,255,0.25)" }}>
              Train an agent first to run simulation.
            </div>
          )}

          {simError && (
            <div
              className="px-4 py-2 rounded text-xs font-mono"
              style={{
                background: "rgba(255,68,102,0.06)",
                border: "1px solid rgba(255,68,102,0.2)",
                color: "#ff4466",
              }}
            >
              {simError}
            </div>
          )}

          {simResult && (
            <div className="space-y-6">
              <Stats stats={simResult.stats} />

              <div>
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-1 h-4 rounded" style={{ background: "#00ffff", boxShadow: "0 0 4px #00ffff" }} />
                  <h3
                    className="text-xs font-bold tracking-widest uppercase font-mono"
                    style={{ color: "rgba(0,255,255,0.6)" }}
                  >
                    Price & Portfolio
                  </h3>
                  <span
                    className="text-xs font-mono ml-2 px-2 py-0.5 rounded"
                    style={{
                      background: "rgba(0,255,255,0.06)",
                      border: "1px solid rgba(0,255,255,0.12)",
                      color: "rgba(0,255,255,0.5)",
                    }}
                  >
                    {ticker} · {algoLabel} · {periodLabel}
                  </span>
                </div>
                <Chart
                  prices={simResult.prices}
                  portfolioValues={simResult.portfolio_values}
                  buySignals={simResult.buy_signals}
                  sellSignals={simResult.sell_signals}
                />
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <div
                  className="p-3 rounded text-xs font-mono"
                  style={{
                    background: "rgba(0,255,136,0.05)",
                    border: "1px solid rgba(0,255,136,0.12)",
                  }}
                >
                  <div style={{ color: "rgba(255,255,255,0.3)" }} className="mb-1 uppercase tracking-wider">
                    Buy Signals
                  </div>
                  <div style={{ color: "#00ff88" }}>{simResult.buy_signals.length}</div>
                </div>
                <div
                  className="p-3 rounded text-xs font-mono"
                  style={{
                    background: "rgba(255,68,102,0.05)",
                    border: "1px solid rgba(255,68,102,0.12)",
                  }}
                >
                  <div style={{ color: "rgba(255,255,255,0.3)" }} className="mb-1 uppercase tracking-wider">
                    Sell Signals
                  </div>
                  <div style={{ color: "#ff4466" }}>{simResult.sell_signals.length}</div>
                </div>
                <div
                  className="p-3 rounded text-xs font-mono"
                  style={{
                    background: "rgba(167,139,250,0.05)",
                    border: "1px solid rgba(167,139,250,0.12)",
                  }}
                >
                  <div style={{ color: "rgba(255,255,255,0.3)" }} className="mb-1 uppercase tracking-wider">
                    Data Points
                  </div>
                  <div style={{ color: "#a78bfa" }}>{simResult.prices.length}</div>
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
