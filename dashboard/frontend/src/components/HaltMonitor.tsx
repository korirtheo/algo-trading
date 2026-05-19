import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';

const ACTION_COLORS: Record<string, string> = {
  TRADED: 'var(--green)',
  OPEN: '#4a7cbe',
  WATCHING: '#aaaa44',
  SKIPPED: '#555',
};

export function HaltMonitor() {
  const { data: events } = usePolling(api.halts, 5000);

  if (!events || events.length === 0) return null;

  return (
    <div className="card animate-in" style={{ gridColumn: '1 / -1' }}>
      <div className="card-header">
        <span className="card-title">Halt Monitor</span>
        <span className="card-subtitle">intraday halt-resume scanner</span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Resume Time</th>
              <th style={{ textAlign: 'left' }}>Ticker</th>
              <th style={{ textAlign: 'left' }}>Reason</th>
              <th style={{ textAlign: 'right' }}>Halt</th>
              <th style={{ textAlign: 'right' }}>Resume</th>
              <th style={{ textAlign: 'left' }}>Action</th>
              <th style={{ textAlign: 'right' }}>PnL</th>
              <th style={{ textAlign: 'left' }}>Exit</th>
            </tr>
          </thead>
          <tbody>
            {events.map((ev, i) => {
              const resumeTime = ev.resume_ts ? ev.resume_ts.slice(11, 19) : (ev.halt_time || '');
              const pnlColor =
                ev.pnl == null
                  ? 'var(--text-secondary)'
                  : ev.pnl >= 0 ? 'var(--green)' : 'var(--red)';
              return (
                <tr key={`${ev.ticker}-${i}`}>
                  <td style={{ fontFamily: 'monospace' }}>{resumeTime}</td>
                  <td style={{ fontWeight: 700 }}>{ev.ticker}</td>
                  <td>
                    <span
                      style={{
                        background: '#333', color: '#ccc',
                        padding: '2px 6px', borderRadius: 3, fontSize: 10, fontWeight: 600,
                      }}
                    >
                      {ev.reason || '?'}
                    </span>
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    {ev.halt_price != null ? `$${ev.halt_price.toFixed(2)}` : '—'}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    {ev.resume_price != null ? `$${ev.resume_price.toFixed(2)}` : '—'}
                  </td>
                  <td>
                    <span style={{
                      background: ACTION_COLORS[ev.action] || '#444',
                      color: '#fff',
                      padding: '2px 8px',
                      borderRadius: 4,
                      fontSize: 10,
                      fontWeight: 700,
                    }}>
                      {ev.action}
                    </span>
                  </td>
                  <td style={{ textAlign: 'right', color: pnlColor, fontWeight: 700 }}>
                    {ev.pnl != null ? `$${ev.pnl >= 0 ? '+' : ''}${ev.pnl.toFixed(2)}` : '—'}
                  </td>
                  <td style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    {ev.action === 'TRADED' && ev.entry_price != null && ev.exit_price != null
                      ? `$${ev.entry_price.toFixed(2)} → $${ev.exit_price.toFixed(2)}`
                      : '—'}
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
