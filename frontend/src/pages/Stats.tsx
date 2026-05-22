interface StatsData {
  final_portfolio_value: number;
  total_profit: number;
  total_profit_pct: number;
  num_trades: number;
  initial_balance: number;
}

interface StatsProps {
  stats: StatsData;
}

export default function Stats({ stats }: StatsProps) {
  const isProfit = stats.total_profit >= 0;

  const items = [
    {
      label: "Final Portfolio",
      value: `$${stats.final_portfolio_value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      color: "#00ffff",
      glow: "rgba(0,255,255,0.3)",
    },
    {
      label: "Total Profit / Loss",
      value: `${isProfit ? "+" : ""}$${stats.total_profit.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      color: isProfit ? "#00ff88" : "#ff4466",
      glow: isProfit ? "rgba(0,255,136,0.3)" : "rgba(255,68,102,0.3)",
    },
    {
      label: "Return %",
      value: `${isProfit ? "+" : ""}${stats.total_profit_pct.toFixed(2)}%`,
      color: isProfit ? "#00ff88" : "#ff4466",
      glow: isProfit ? "rgba(0,255,136,0.3)" : "rgba(255,68,102,0.3)",
    },
    {
      label: "Total Trades",
      value: stats.num_trades.toString(),
      color: "#a78bfa",
      glow: "rgba(167,139,250,0.3)",
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map((item) => (
        <div
          key={item.label}
          className="stat-card p-4 flex flex-col gap-1"
          style={{
            background: "rgba(13,17,23,0.9)",
            border: `1px solid ${item.color}22`,
          }}
        >
          <span
            className="text-xs font-mono uppercase tracking-widest"
            style={{ color: "rgba(255,255,255,0.35)" }}
          >
            {item.label}
          </span>
          <span
            className="text-xl font-bold font-mono"
            style={{
              color: item.color,
              textShadow: `0 0 8px ${item.glow}`,
            }}
          >
            {item.value}
          </span>
        </div>
      ))}
    </div>
  );
}
