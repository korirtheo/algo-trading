import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';
import { strategyColor } from './StrategyTag';

const STATUS_LABEL: Record<string, string> = {
  fired: 'Fired',
  watching: 'Watching',
  timed_out: 'Timed Out',
  not_eligible: 'Not Eligible',
};

export function Diagnostics() {
  const { data: items } = usePolling(api.diagnostics, 5000);

  if (!items || items.length === 0) return null;

  return (
    <div className="card animate-in" style={{ gridColumn: '1 / -1' }}>
      <div className="card-header">
        <span className="card-title">Signal Diagnostics</span>
        <span className="card-subtitle">why each ticker did / didn't trade</span>
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, padding: '6px 12px', borderBottom: '1px solid var(--border)', fontSize: 11, color: 'var(--text-secondary)' }}>
        <span><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: 'var(--green)', marginRight: 4 }} />Fired</span>
        <span><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: '#4a7cbe', marginRight: 4 }} />Watching</span>
        <span><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: '#555', opacity: 0.6, marginRight: 4 }} />Timed Out</span>
        <span><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: '#333', marginRight: 4 }} />None Eligible</span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Ticker</th>
              <th style={{ textAlign: 'right' }}>Price</th>
              <th style={{ textAlign: 'right' }}>Chg</th>
              <th style={{ textAlign: 'right' }}>Gap</th>
              <th style={{ textAlign: 'right' }}>Candles</th>
              <th style={{ textAlign: 'right' }}>Open</th>
              <th style={{ textAlign: 'right' }}>PM High</th>
              <th style={{ textAlign: 'right' }}>PM High vs Open</th>
              <th style={{ textAlign: 'left', minWidth: 120 }}>Strategies</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.ticker}>
                <td style={{ fontWeight: 700 }}>
                  {item.ticker}
                  {item.traded && (
                    <span style={{ marginLeft: 6, fontSize: 10, color: 'var(--green)', fontWeight: 400 }}>TRADED</span>
                  )}
                </td>
                <td style={{ textAlign: 'right', fontWeight: 600 }}>
                  {item.last_price != null ? `$${item.last_price.toFixed(2)}` : '—'}
                </td>
                <td style={{ textAlign: 'right', fontWeight: 700, color: item.change_pct == null ? 'var(--text-secondary)' : item.change_pct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {item.change_pct != null ? `${item.change_pct >= 0 ? '+' : ''}${item.change_pct.toFixed(1)}%` : '—'}
                </td>
                <td style={{ textAlign: 'right', color: 'var(--green)' }}>+{item.gap_pct.toFixed(0)}%</td>
                <td style={{ textAlign: 'right' }}>{item.candle_count}</td>
                <td style={{ textAlign: 'right' }}>${item.market_open.toFixed(3)}</td>
                <td style={{ textAlign: 'right' }}>${item.premarket_high.toFixed(3)}</td>
                <td style={{ textAlign: 'right', color: item.pm_high_pct_above_open > 5 ? 'var(--red)' : 'var(--text-secondary)' }}>
                  +{item.pm_high_pct_above_open.toFixed(1)}%
                </td>
                <td>
                  <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                    {item.strategies.filter((s) => s.status !== 'not_eligible').length === 0 ? (
                      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>none</span>
                    ) : (
                      item.strategies.filter((s) => s.status !== 'not_eligible').map((s) => {
                        const color = s.status === 'timed_out' ? '#444'
                          : s.status === 'fired' ? strategyColor(s.code)
                          : strategyColor(s.code) + '88';
                        const textColor = s.status === 'timed_out' ? '#777' : '#fff';
                        return (
                          <span
                            key={s.code}
                            title={`${s.code}: ${STATUS_LABEL[s.status]}`}
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              width: 22,
                              height: 22,
                              borderRadius: 4,
                              fontSize: 10,
                              fontWeight: 700,
                              background: color,
                              color: textColor,
                              outline: s.status === 'fired' ? '1.5px solid #fff' : 'none',
                              textDecoration: s.status === 'timed_out' ? 'line-through' : 'none',
                            }}
                          >
                            {s.code}
                          </span>
                        );
                      })
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
