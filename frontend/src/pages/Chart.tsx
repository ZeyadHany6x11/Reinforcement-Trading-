import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface ChartProps {
  prices: number[];
  portfolioValues: number[];
  buySignals: number[];
  sellSignals: number[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div
        style={{
          background: "rgba(13,17,23,0.95)",
          border: "1px solid rgba(0,255,255,0.25)",
          borderRadius: "6px",
          padding: "10px 14px",
          fontSize: "12px",
          fontFamily: "monospace",
        }}
      >
        <p style={{ color: "rgba(255,255,255,0.4)", marginBottom: 4 }}>Step {label}</p>
        {payload.map((p: any) => (
          <p key={p.dataKey} style={{ color: p.color, margin: "2px 0" }}>
            {p.name}: ${Number(p.value).toFixed(2)}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function Chart({ prices, portfolioValues, buySignals, sellSignals }: ChartProps) {
  const data = prices.map((price, i) => ({
    step: i,
    Price: parseFloat(price.toFixed(2)),
    Portfolio: parseFloat((portfolioValues[i + 1] ?? portfolioValues[i]).toFixed(2)),
    isBuy: buySignals.includes(i),
    isSell: sellSignals.includes(i),
  }));

  return (
    <div className="w-full">
      <div className="flex gap-4 mb-4 flex-wrap">
        <div className="flex items-center gap-2 text-xs font-mono">
          <div className="w-4 h-px" style={{ background: "#00ffff", boxShadow: "0 0 4px #00ffff" }} />
          <span style={{ color: "rgba(0,255,255,0.7)" }}>Price</span>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <div className="w-4 h-px" style={{ background: "#00ff88", boxShadow: "0 0 4px #00ff88" }} />
          <span style={{ color: "rgba(0,255,136,0.7)" }}>Portfolio</span>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <div
            className="w-3 h-3 rounded-full"
            style={{ background: "#00ff88", boxShadow: "0 0 6px #00ff88" }}
          />
          <span style={{ color: "rgba(0,255,136,0.7)" }}>Buy</span>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <div
            className="w-3 h-3 rounded-full"
            style={{ background: "#ff4466", boxShadow: "0 0 6px #ff4466" }}
          />
          <span style={{ color: "rgba(255,68,102,0.7)" }}>Sell</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,255,255,0.06)" />
          <XAxis
            dataKey="step"
            tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10, fontFamily: "monospace" }}
            tickLine={false}
            axisLine={{ stroke: "rgba(0,255,255,0.1)" }}
            interval={Math.floor(data.length / 8)}
          />
          <YAxis
            yAxisId="price"
            tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10, fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
            width={60}
          />
          <YAxis
            yAxisId="portfolio"
            orientation="right"
            tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10, fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
            width={60}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="Price"
            stroke="#00ffff"
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 4, fill: "#00ffff", strokeWidth: 0 }}
          />
          <Line
            yAxisId="portfolio"
            type="monotone"
            dataKey="Portfolio"
            stroke="#00ff88"
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 4, fill: "#00ff88", strokeWidth: 0 }}
          />
          {buySignals.map((idx) => (
            <ReferenceLine
              key={`buy-${idx}`}
              yAxisId="price"
              x={idx}
              stroke="rgba(0,255,136,0.35)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
          ))}
          {sellSignals.map((idx) => (
            <ReferenceLine
              key={`sell-${idx}`}
              yAxisId="price"
              x={idx}
              stroke="rgba(255,68,102,0.35)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
