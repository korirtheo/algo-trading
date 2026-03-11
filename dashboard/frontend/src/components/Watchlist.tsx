import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';
import { strategyColor } from './StrategyTag';

interface Props {
  onSelectSymbol: (symbol: string) => void;
  selectedSymbol: string | null;
}

export function Watchlist({ onSelectSymbol, selectedSymbol }: Props) {
  const { data: items } = usePolling(api.watchlist, 5000);

  return (
    <div className="card animate-in">
      <div className="card-header">
        <span className="card-title">Watchlist</span>
        <span className="card-subtitle">{items?.length || 0} tickers</span>
      </div>
      <div style={{ maxHeight: 420, overflowY: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Ticker</th>
              <th style={{ textAlign: 'right' }}>Price</th>
              <th style={{ textAlign: 'right' }}>Chg</th>
              <th style={{ textAlign: 'right' }}>PM Vol</th>
              <th style={{ textAlign: 'right' }}>Float</th>
              <th style={{ textAlign: 'center' }}>Status</th>
              <th style={{ textAlign: 'center' }}>Strat</th>
            </tr>
          </thead>
          <tbody>
            {(items || []).map((item) => {
              const chg = item.change_pct;
              const chgColor = chg == null ? 'var(--text-secondary)' : chg >= 0 ? 'var(--green)' : 'var(--red)';
              return (
                <tr
                  key={item.ticker}
                  className={`watch-row ${selectedSymbol === item.ticker ? 'selected' : ''}`}
                  onClick={() => onSelectSymbol(item.ticker)}
                >
                  <td className="ticker-link">{item.ticker}</td>
                  <td style={{ textAlign: 'right', fontWeight: 600 }}>
                    {item.last_price != null ? `$${item.last_price.toFixed(2)}` : '—'}
                  </td>
                  <td style={{ textAlign: 'right', color: chgColor, fontWeight: 700 }}>
                    {chg != null ? `${chg >= 0 ? '+' : ''}${chg.toFixed(1)}%` : `+${item.gap_pct.toFixed(0)}%`}
                  </td>
                  <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>
                    {(item.pm_volume / 1000).toFixed(0)}K
                  </td>
                  <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>
                    {item.float_shares ? `${(item.float_shares / 1e6).toFixed(1)}M` : '—'}
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    <span className={`status-badge status-${item.status}`}>
                      {item.status.replace('_', ' ')}
                    </span>
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    {item.strategy && (
                      <span
                        className="strat-tag"
                        style={{
                          width: 22,
                          height: 22,
                          fontSize: 10,
                          background: `linear-gradient(135deg, ${strategyColor(item.strategy)}, ${strategyColor(item.strategy)}cc)`,
                        }}
                      >
                        {item.strategy}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
