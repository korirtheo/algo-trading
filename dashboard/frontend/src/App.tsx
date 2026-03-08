import { useState } from 'react';
import { PortfolioHeader } from './components/PortfolioHeader';
import { Watchlist } from './components/Watchlist';
import { Chart } from './components/Chart';
import { Positions } from './components/Positions';
import { TradeLog } from './components/TradeLog';
import { StrategyPanel } from './components/StrategyPanel';
import { useWebSocket } from './hooks/useWebSocket';

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const { connected } = useWebSocket('/ws/live');

  return (
    <div className="app-root">
      {/* ─── Top Navigation Bar ─── */}
      <header className="top-bar">
        <div className="top-bar-left">
          <div className="logo-mark" />
          <span className="logo-text">AlgoTrader</span>
          <span className="badge badge-mode">PAPER</span>
        </div>
        <div className="top-bar-right">
          <div className="connection-status">
            <span className={`status-dot ${connected ? 'live' : 'offline'}`} />
            <span className={`status-label ${connected ? 'live' : 'offline'}`}>
              {connected ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>
          <div className="header-divider" />
          <span className="header-meta">12 Strategies</span>
          <span className="badge badge-trial">T-432</span>
        </div>
      </header>

      {/* ─── Dashboard Grid ─── */}
      <main className="dashboard-main">
        <PortfolioHeader />

        <div className="row-chart-watch">
          <Chart symbol={selectedSymbol} />
          <Watchlist onSelectSymbol={setSelectedSymbol} selectedSymbol={selectedSymbol} />
        </div>

        <div className="row-bottom">
          <Positions />
          <TradeLog />
          <StrategyPanel />
        </div>
      </main>
    </div>
  );
}

export default App;
