import { api, DailyReconcile } from '../api/client';
import { usePolling } from '../hooks/usePolling';

function StatusBadge({ status }: { status: string }) {
  const cls = status === 'match' ? 'status-ok'
    : status === 'divergence' || status === 'live_only_missed_by_replay' ? 'status-error'
    : status === 'no_trades' ? 'status-warn'
    : 'status-warn';
  return <span className={`status-badge ${cls}`}>{status.replace(/_/g, ' ')}</span>;
}

function TickerList({ label, items, color }: { label: string; items: string[]; color: string }) {
  if (!items?.length) return null;
  return (
    <div style={{ marginBottom: 8 }}>
      <span style={{ color, fontWeight: 700, marginRight: 8 }}>{label}:</span>
      <span style={{ color: 'var(--text-secondary)' }}>{items.join(', ')}</span>
    </div>
  );
}

export function Reconcile() {
  const { data: rows } = usePolling<DailyReconcile[]>(api.reconcile, 30000);

  return (
    <div className="card animate-in">
      <div className="card-header">
        <span className="card-title">Daily Reconcile</span>
        <span className="card-subtitle">
          {rows?.length ? `last run: ${rows[0].date}` : 'no runs yet'}
        </span>
      </div>
      <div style={{ maxHeight: 500, overflowY: 'auto' }}>
        {(rows?.length || 0) === 0 ? (
          <div style={{ padding: 16, color: 'var(--text-secondary)' }}>
            No reconcile runs yet. Runs automatically after 8pm ET extended hours.
          </div>
        ) : (
          rows?.map((r) => (
            <div
              key={r.id}
              style={{
                borderBottom: '1px solid rgba(255,255,255,0.06)',
                padding: '10px 4px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <strong style={{ fontSize: 14 }}>{r.date}</strong>
                  <span style={{ color: 'var(--text-secondary)', fontSize: 11, marginLeft: 8 }}>
                    watchlist {r.watchlist_count} · SIP {r.sip_fetched}
                  </span>
                </div>
                <StatusBadge status={r.status} />
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>
                live: {r.live_trades} fills (${(r.live_pnl || 0).toFixed(0)}) · backtest: {r.bt_trades} (${(r.bt_pnl || 0).toFixed(0)}) · match {r.match_count} · live-only {r.live_only_count} · bt-only {r.bt_only_count}
              </div>
              {r.summary && (
                <div style={{ fontSize: 12, marginTop: 4, color: 'var(--text-primary)' }}>{r.summary}</div>
              )}
              {r.details?.comments?.map((c, i) => (
                <div key={i} style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2, paddingLeft: 8, borderLeft: '2px solid rgba(255,255,255,0.12)' }}>
                  {c}
                </div>
              ))}
              <TickerList label="MATCH" items={(r.match_tickers || '').split(',').filter(Boolean)} color="var(--green)" />
              <TickerList label="LIVE-ONLY" items={(r.live_only_tickers || '').split(',').filter(Boolean)} color="var(--red)" />
              <TickerList label="BT-ONLY" items={(r.bt_only_tickers || '').split(',').filter(Boolean)} color="#ffa500" />
            </div>
          ))
        )}
      </div>
    </div>
  );
}
