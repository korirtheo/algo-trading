import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';
import { StrategyTag } from './StrategyTag';

export function Positions() {
  const { data: positions } = usePolling(api.positions, 3000);

  return (
    <div className="card animate-in">
      <div className="card-header">
        <span className="card-title">Open Positions</span>
        <span className="card-subtitle">{positions?.length || 0} active</span>
      </div>
      {(!positions || positions.length === 0) ? (
        <div className="empty-state">No open positions</div>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Ticker</th>
              <th style={{ textAlign: 'center' }}>Strat</th>
              <th style={{ textAlign: 'right' }}>Qty</th>
              <th style={{ textAlign: 'right' }}>Entry</th>
              <th style={{ textAlign: 'right' }}>Current</th>
              <th style={{ textAlign: 'right' }}>P&L</th>
              <th style={{ textAlign: 'right' }}>%</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => {
              const cls = p.unrealized_pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
              return (
                <tr key={p.ticker}>
                  <td className="ticker-link">{p.ticker}</td>
                  <td style={{ textAlign: 'center' }}><StrategyTag code={p.strategy} /></td>
                  <td style={{ textAlign: 'right' }}>{p.qty.toFixed(0)}</td>
                  <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>${p.avg_entry.toFixed(2)}</td>
                  <td style={{ textAlign: 'right' }}>${p.current_price.toFixed(2)}</td>
                  <td style={{ textAlign: 'right', fontWeight: 700 }} className={cls}>
                    {p.unrealized_pnl >= 0 ? '+' : ''}${p.unrealized_pnl.toFixed(2)}
                  </td>
                  <td style={{ textAlign: 'right' }} className={cls}>
                    {p.unrealized_pnl_pct >= 0 ? '+' : ''}{p.unrealized_pnl_pct.toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
