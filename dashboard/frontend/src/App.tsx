import { useState } from 'react';
import { PortfolioHeader } from './components/PortfolioHeader';
import { Watchlist } from './components/Watchlist';
import { Chart } from './components/Chart';
import { Positions } from './components/Positions';
import { TradeLog } from './components/TradeLog';
import { StrategyPanel } from './components/StrategyPanel';
import { Diagnostics } from './components/Diagnostics';
import { HaltMonitor } from './components/HaltMonitor';
import { Slippage } from './components/Slippage';
import { SystemHealth } from './components/SystemHealth';
import { Analytics } from './components/Analytics';
import { useWebSocket } from './hooks/useWebSocket';

type Page = 'dashboard' | 'analytics';

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [page, setPage] = useState<Page>('dashboard');
  const { connected } = useWebSocket('/ws/live');

  return (
    <div className="app-root">
      {/* ─── Top Navigation Bar ─── */}
      <header className="top-bar">
        <div className="top-bar-left">
          <div className="logo-mark" />
          <span className="logo-text">AlgoTrader</span>
          <span className="badge badge-mode">PAPER</span>
          <nav className="page-nav">
            <button
              className={`page-nav-btn ${page === 'dashboard' ? 'active' : ''}`}
              onClick={() => setPage('dashboard')}
            >
              Dashboard
            </button>
            <button
              className={`page-nav-btn ${page === 'analytics' ? 'active' : ''}`}
              onClick={() => setPage('analytics')}
            >
              Analytics
            </button>
          </nav>
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

      {/* ─── Dashboard Grid or Analytics Page ─── */}
      <main className="dashboard-main">
        {page === 'dashboard' ? (
          <>
            <SystemHealth />
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

            <div className="row-diagnostics">
              <Diagnostics />
            </div>

            <div className="row-diagnostics">
              <HaltMonitor />
            </div>

            <div className="row-diagnostics">
              <Slippage />
            </div>
          </>
        ) : (
          <Analytics />
        )}
      </main>
    </div>
  );
}

export default App;
