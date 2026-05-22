import { useEffect, useRef } from "react";

interface HeroProps {
  onEnter: () => void;
}

export default function Hero({ onEnter }: HeroProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles: { x: number; y: number; vx: number; vy: number; size: number; alpha: number }[] = [];
    for (let i = 0; i < 80; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        size: Math.random() * 2 + 0.5,
        alpha: Math.random() * 0.6 + 0.1,
      });
    }

    let animId: number;
    function draw() {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0,255,255,${p.alpha})`;
        ctx.fill();
      });

      particles.forEach((p, i) => {
        particles.slice(i + 1).forEach((q) => {
          const dx = p.x - q.x;
          const dy = p.y - q.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.strokeStyle = `rgba(0,255,255,${0.12 * (1 - dist / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        });
      });

      animId = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(animId);
  }, []);

  return (
    <div
      className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden"
      style={{ background: "#060b10" }}
    >
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />

      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 50%, rgba(0,255,255,0.04) 0%, rgba(6,11,16,0.85) 70%)",
        }}
      />

      <div className="relative z-10 flex flex-col items-center text-center px-6 max-w-3xl">
        <div
          className="mb-4 text-xs font-mono tracking-[0.3em] uppercase"
          style={{ color: "rgba(0,255,255,0.5)" }}
        >
          — Powered by AI —
        </div>

        <h1
          className="font-black uppercase leading-none mb-2"
          style={{
            fontSize: "clamp(2.8rem, 8vw, 5.5rem)",
            color: "#fff",
            letterSpacing: "-0.01em",
            textShadow: "0 0 60px rgba(0,255,255,0.15)",
          }}
        >
          REINFORCEMENT
        </h1>
        <h1
          className="font-black uppercase leading-none mb-2"
          style={{
            fontSize: "clamp(2.8rem, 8vw, 5.5rem)",
            color: "#fff",
            letterSpacing: "-0.01em",
            textShadow: "0 0 60px rgba(0,255,255,0.15)",
          }}
        >
          LEARNING
        </h1>
        <h2
          className="font-bold uppercase tracking-[0.25em] mb-8"
          style={{
            fontSize: "clamp(1.2rem, 4vw, 2.2rem)",
            color: "rgba(255,255,255,0.55)",
            letterSpacing: "0.3em",
          }}
        >
          TRADING
        </h2>

        <div
          className="mb-10 px-6 py-2 rounded-full text-sm font-mono"
          style={{
            border: "1px solid rgba(0,255,255,0.25)",
            color: "rgba(0,255,255,0.8)",
            background: "rgba(0,255,255,0.04)",
            letterSpacing: "0.05em",
          }}
        >
          AI Adaptive Strategies for Real-Time Markets
        </div>

        <div className="flex gap-4 mb-10 flex-wrap justify-center">
          {["Q-Learning", "Deep Q-Network"].map((algo) => (
            <span
              key={algo}
              className="text-xs font-mono px-3 py-1 rounded"
              style={{
                background: "rgba(0,255,255,0.07)",
                border: "1px solid rgba(0,255,255,0.18)",
                color: "rgba(0,255,255,0.65)",
              }}
            >
              {algo}
            </span>
          ))}
          {["AAPL", "BTC-USD", "TSLA", "SPY"].map((t) => (
            <span
              key={t}
              className="text-xs font-mono px-3 py-1 rounded"
              style={{
                background: "rgba(0,255,100,0.06)",
                border: "1px solid rgba(0,255,100,0.15)",
                color: "rgba(0,255,100,0.6)",
              }}
            >
              {t}
            </span>
          ))}
        </div>

        <button
          onClick={onEnter}
          className="relative group px-10 py-4 font-bold text-sm tracking-widest uppercase rounded transition-all duration-300"
          style={{
            background: "rgba(0,255,255,0.1)",
            border: "1px solid #00ffff",
            color: "#00ffff",
            boxShadow: "0 0 20px rgba(0,255,255,0.3), 0 0 60px rgba(0,255,255,0.08)",
            letterSpacing: "0.2em",
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.boxShadow =
              "0 0 30px rgba(0,255,255,0.6), 0 0 80px rgba(0,255,255,0.2)";
            (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,255,255,0.18)";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.boxShadow =
              "0 0 20px rgba(0,255,255,0.3), 0 0 60px rgba(0,255,255,0.08)";
            (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,255,255,0.1)";
          }}
        >
          ENTER DASHBOARD
        </button>
      </div>

      <div
        className="absolute bottom-8 left-1/2 -translate-x-1/2 font-mono text-xs tracking-widest"
        style={{ color: "rgba(0,255,255,0.25)" }}
      >
        Q-LEARNING • DQN • LIVE DATA
      </div>
    </div>
  );
}
