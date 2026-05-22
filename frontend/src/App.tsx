import { useState } from "react";
import Hero from "@/pages/Hero";
import Dashboard from "@/pages/Dashboard";

function App() {
  const [showDashboard, setShowDashboard] = useState(false);

  if (showDashboard) {
    return <Dashboard />;
  }

  return <Hero onEnter={() => setShowDashboard(true)} />;
}

export default App;
