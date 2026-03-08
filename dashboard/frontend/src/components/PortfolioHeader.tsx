import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';

function StatCard({ label, value, sub, color, accent }: {
  label: string; value: string; sub?: string; color?: string; accent?: string;
}) {
  return (
    <div className={`stat-card ${accent ? `accent-${accent}` : 'accent-blue'}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color: color || 'var(--text-primary)' }}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export function PortfolioHeader() {
  const { data: account } = usePolling(api.account, 5000);
  const { data: summary } = usePolling(api.summary, 5000);

  const a = account || { cash: 0, buying_power: 0, equity: 0, portfolio_value: 0, daily_pnl: 0, status: '...', pdt: false, daytrade_count: 0 };
  const s = summary || { active_position: null, daily_pnl: 0, trades_count: 0, wins: 0, losses: 0, candidates_count: 0, tracking_count: 0 };

  const pnlColor = a.daily_pnl >= 0 ? 'var(--green)' : 'var(--red)';
  const wr = s.trades_count > 0 ? ((s.wins / s.trades_count) * 100).toFixed(0) : '—';

  return (
    <div className="stats-grid animate-in">
      <StatCard
        label="Portfolio Value"
        value={`$${a.portfolio_value?.toLocaleString(undefined, { minimumFractionDigits: 2 }) || '0.00'}`}
        sub={a.status}
        accent="blue"
      />
      <StatCard
        label="Daily P&L"
        value={`${a.daily_pnl >= 0 ? '+' : ''}$${a.daily_pnl?.toLocaleString(undefined, { minimumFractionDigits: 2 }) || '0.00'}`}
        color={pnlColor}
        sub={`Engine: ${s.daily_pnl >= 0 ? '+' : ''}$${s.daily_pnl?.toFixed(0)}`}
        accent={a.daily_pnl >= 0 ? 'green' : 'red'}
      />
      <StatCard
        label="Cash"
        value={`$${a.cash?.toLocaleString(undefined, { minimumFractionDigits: 0 }) || '0'}`}
        sub={`BP: $${a.buying_power?.toLocaleString(undefined, { minimumFractionDigits: 0 })}`}
        accent="blue"
      />
      <StatCard
        label="Trades Today"
        value={`${s.trades_count}`}
        sub={`${s.wins}W / ${s.losses}L (${wr}%)`}
        color={s.wins >= s.losses ? 'var(--green)' : 'var(--red)'}
        accent={s.wins >= s.losses ? 'green' : 'red'}
      />
      <StatCard
        label="Active Position"
        value={s.active_position || 'None'}
        color={s.active_position ? 'var(--blue-bright)' : 'var(--text-muted)'}
        sub={`${s.candidates_count} candidates`}
        accent="purple"
      />
    </div>
  );
}
