import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';
import { StrategyTag } from './StrategyTag';

function reasonClass(reason: string): string {
  if (reason === 'TARGET' || reason === 'TRAIL') return 'reason-win';
  if (reason === 'STOP') return 'reason-loss';
  return 'reason-neutral';
}

export function TradeLog() {
  const { data: trades } = usePolling(api.trades, 5000);

  return (
    <div className="card animate-in">
      <div className="card-header">
        <span className="card-title">Trade Log</span>
        <span className="card-subtitle">{trades?.length || 0} trades today</span>
      </div>
      {(!trades || trades.length === 0) ? (
        <div className="empty-state">No trades yet today</div>
      ) : (
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Ticker</th>
                <th style={{ textAlign: 'center' }}>Strat</th>
                <th style={{ textAlign: 'right' }}>Entry</th>
                <th style={{ textAlign: 'right' }}>Exit</th>
                <th style={{ textAlign: 'right' }}>P&L</th>
                <th style={{ textAlign: 'right' }}>%</th>
                <th style={{ textAlign: 'center' }}>Reason</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t, i) => {
                const cls = t.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                return (
                  <tr key={i}>
                    <td className="ticker-link">{t.ticker}</td>
                    <td style={{ textAlign: 'center' }}><StrategyTag code={t.strategy} size={20} /></td>
                    <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>${t.entry_price.toFixed(2)}</td>
                    <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>${t.exit_price.toFixed(2)}</td>
                    <td style={{ textAlign: 'right', fontWeight: 700 }} className={cls}>
                      {t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(0)}
                    </td>
                    <td style={{ textAlign: 'right' }} className={cls}>
                      {t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct.toFixed(1)}%
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <span className={`reason-badge ${reasonClass(t.reason)}`}>{t.reason}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
