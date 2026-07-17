import { useState } from 'react';
import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';

function fmtBp(v: number | null): string {
  if (v === null || v === undefined) return '—';
  return `${v >= 0 ? '+' : ''}${v.toFixed(1)} bp`;
}

function fmtPct(v: number | null): string {
  if (v === null || v === undefined) return '—';
  return `${(v * 100).toFixed(3)}%`;
}

function bpColor(v: number | null): string {
  if (v === null || v === undefined) return 'var(--text-secondary)';
  if (v > 20) return 'var(--red)';
  if (v < -10) return 'var(--green)';
  return 'var(--text-secondary)';
}

function fmtTime(ts: string): string {
  if (!ts) return '';
  const t = ts.split('T')[1];
  return t ? t.slice(0, 8) : ts;
}

const ROWS_PER_PAGE = 15;

export function Slippage() {
  const { data } = usePolling(api.slippage, 5000);
  const [page, setPage] = useState(0);

  if (!data) return null;
  const { stats, by_strategy, rows } = data;

  const totalPages = Math.ceil(rows.length / ROWS_PER_PAGE);
  const paginatedRows = rows.slice(page * ROWS_PER_PAGE, (page + 1) * ROWS_PER_PAGE);

  return (
    <div className="card animate-in" style={{ gridColumn: '1 / -1' }}>
      <div className="card-header">
        <span className="card-title">Slippage Calibration</span>
        <span className="card-subtitle">
          live fill vs signal price — used to calibrate the cost model
        </span>
      </div>

      {/* Summary stats row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
        gap: 12,
        padding: '12px',
        borderBottom: '1px solid var(--border)',
        fontSize: 12,
      }}>
        <Stat label="Fills" value={`${stats.n_filled} / ${stats.n_total}`} />
        <Stat label="Avg slip" value={fmtBp(stats.avg_slip_bp)} color={bpColor(stats.avg_slip_bp)} />
        <Stat label="Median slip" value={fmtBp(stats.median_slip_bp)} color={bpColor(stats.median_slip_bp)} />
        <Stat label="p95 slip" value={fmtBp(stats.p95_slip_bp)} color={bpColor(stats.p95_slip_bp)} />
        <Stat label="Best slip" value={fmtBp(stats.min_slip_bp)} color={bpColor(stats.min_slip_bp)} />
        <Stat label="Worst slip" value={fmtBp(stats.max_slip_bp)} color={bpColor(stats.max_slip_bp)} />
        <Stat label="$ volume" value={`$${stats.dollar_volume_traded.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
        <Stat
          label="Realized cost"
          value={`$${stats.realized_slip_cost.toFixed(2)}`}
          color={stats.realized_slip_cost > 0 ? 'var(--red)' : 'var(--green)'}
        />
      </div>

      {/* Per-strategy breakdown */}
      {by_strategy && by_strategy.length > 0 && (
        <div style={{ borderBottom: '1px solid var(--border)' }}>
          <div style={{
            fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase',
            letterSpacing: 0.5, padding: '8px 12px 4px',
          }}>
            By Strategy
          </div>
          <table className="data-table" style={{ marginBottom: 4 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Strat</th>
                <th style={{ textAlign: 'right' }}>Fills</th>
                <th style={{ textAlign: 'right' }}>Avg</th>
                <th style={{ textAlign: 'right' }}>Median</th>
                <th style={{ textAlign: 'right' }}>Avg Buy</th>
                <th style={{ textAlign: 'right' }}>Avg Sell</th>
                <th style={{ textAlign: 'right' }}>Worst</th>
                <th style={{ textAlign: 'right' }}>$ Volume</th>
                <th style={{ textAlign: 'right' }}>Cost</th>
              </tr>
            </thead>
            <tbody>
              {by_strategy.map((s) => (
                <tr key={s.strategy}>
                  <td style={{ fontWeight: 700 }}>{s.strategy}</td>
                  <td style={{ textAlign: 'right' }}>
                    {s.n}
                    <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                      {' '}({s.n_buys}b/{s.n_sells}s)
                    </span>
                  </td>
                  <td style={{ textAlign: 'right', fontWeight: 700, color: bpColor(s.avg_slip_bp) }}>
                    {fmtBp(s.avg_slip_bp)}
                  </td>
                  <td style={{ textAlign: 'right', color: bpColor(s.median_slip_bp) }}>
                    {fmtBp(s.median_slip_bp)}
                  </td>
                  <td style={{ textAlign: 'right', color: bpColor(s.avg_buy_slip_bp) }}>
                    {fmtBp(s.avg_buy_slip_bp)}
                  </td>
                  <td style={{ textAlign: 'right', color: bpColor(s.avg_sell_slip_bp) }}>
                    {fmtBp(s.avg_sell_slip_bp)}
                  </td>
                  <td style={{ textAlign: 'right', color: bpColor(s.max_slip_bp) }}>
                    {fmtBp(s.max_slip_bp)}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    ${s.dollar_volume.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                  <td style={{
                    textAlign: 'right', fontWeight: 600,
                    color: s.realized_cost > 0 ? 'var(--red)' : 'var(--green)',
                  }}>
                    ${s.realized_cost.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {rows.length === 0 ? (
        <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 13 }}>
          No fills logged yet. Rows will appear after the next live order fills.
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Fill time</th>
                <th style={{ textAlign: 'left' }}>Ticker</th>
                <th style={{ textAlign: 'left' }}>Side</th>
                <th style={{ textAlign: 'left' }}>Strat</th>
                <th style={{ textAlign: 'right' }}>Signal</th>
                <th style={{ textAlign: 'right' }}>Fill</th>
                <th style={{ textAlign: 'right' }}>Slip</th>
                <th style={{ textAlign: 'right' }}>Qty</th>
                <th style={{ textAlign: 'right' }}>$ size</th>
                <th style={{ textAlign: 'right' }}>Particip.</th>
                <th style={{ textAlign: 'left' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {paginatedRows.map((r, i) => {
                const sideColor = r.side === 'buy' ? 'var(--green)' : 'var(--red)';
                return (
                  <tr key={`${r.order_id}-${i}`}>
                    <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{fmtTime(r.ts_fill)}</td>
                    <td style={{ fontWeight: 700 }}>{r.ticker}</td>
                    <td>
                      <span style={{
                        background: sideColor, color: '#fff',
                        padding: '1px 6px', borderRadius: 3,
                        fontSize: 10, fontWeight: 700,
                      }}>{r.side.toUpperCase()}</span>
                    </td>
                    <td style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{r.strategy}</td>
                    <td style={{ textAlign: 'right' }}>
                      {r.signal_price != null ? `$${r.signal_price.toFixed(3)}` : '—'}
                    </td>
                    <td style={{ textAlign: 'right', fontWeight: 600 }}>
                      {r.fill_price != null ? `$${r.fill_price.toFixed(3)}` : '—'}
                    </td>
                    <td style={{ textAlign: 'right', fontWeight: 700, color: bpColor(r.slip_bp) }}>
                      {fmtBp(r.slip_bp)}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      {r.qty != null ? r.qty.toFixed(0) : '—'}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      {r.dollar_amount != null ? `$${r.dollar_amount.toFixed(0)}` : '—'}
                    </td>
                    <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>
                      {fmtPct(r.participation_rate)}
                    </td>
                    <td style={{ fontSize: 11, color: r.status === 'filled' ? 'var(--text-secondary)' : 'var(--red)' }}>
                      {r.status}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {rows.length > ROWS_PER_PAGE && (
        <div style={{ padding: '12px 16px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Page {page + 1} of {totalPages} ({rows.length} fills)
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => setPage(Math.max(0, page - 1))}
              disabled={page === 0}
              style={{
                padding: '4px 12px',
                fontSize: 12,
                fontWeight: 600,
                borderRadius: 4,
                border: '1px solid var(--border)',
                background: page === 0 ? 'var(--bg-card)' : 'var(--bg-elevated)',
                color: page === 0 ? 'var(--text-muted)' : 'var(--text-primary)',
                cursor: page === 0 ? 'not-allowed' : 'pointer'
              }}
            >
              Previous
            </button>
            <button
              onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
              disabled={page === totalPages - 1}
              style={{
                padding: '4px 12px',
                fontSize: 12,
                fontWeight: 600,
                borderRadius: 4,
                border: '1px solid var(--border)',
                background: page === totalPages - 1 ? 'var(--bg-card)' : 'var(--bg-elevated)',
                color: page === totalPages - 1 ? 'var(--text-muted)' : 'var(--text-primary)',
                cursor: page === totalPages - 1 ? 'not-allowed' : 'pointer'
              }}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
        {label}
      </div>
      <div style={{ fontSize: 14, fontWeight: 700, color: color || 'var(--text)' }}>
        {value}
      </div>
    </div>
  );
}
