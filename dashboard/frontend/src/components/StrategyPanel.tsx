import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';
import { StrategyTag } from './StrategyTag';

function catClass(category: string): string {
  if (category === 'momentum') return 'cat-momentum';
  if (category === 'breakout') return 'cat-breakout';
  if (category === 'reversal') return 'cat-reversal';
  if (category === 'squeeze') return 'cat-squeeze';
  return 'cat-default';
}

export function StrategyPanel() {
  const { data: strategies } = usePolling(api.strategies, 10000);
  const { data: config } = usePolling(api.strategyConfig, 30000);

  const configMap = new Map((config || []).map(c => [c.code, c]));

  const sorted = [...(strategies || [])].sort((a, b) => {
    const ac = configMap.get(a.code);
    const bc = configMap.get(b.code);
    const aEnabled = ac?.enabled ?? false;
    const bEnabled = bc?.enabled ?? false;
    if (aEnabled !== bEnabled) return aEnabled ? -1 : 1;
    return (ac?.priority ?? 99) - (bc?.priority ?? 99);
  });

  return (
    <div className="card animate-in">
      <div className="card-header">
        <span className="card-title">Strategy Arsenal</span>
        <span className="card-subtitle">{sorted.filter(s => configMap.get(s.code)?.enabled).length} active</span>
      </div>
      <div style={{ maxHeight: 500, overflowY: 'auto' }}>
        {sorted.map((s) => {
          const cfg = configMap.get(s.code);
          const enabled = cfg?.enabled ?? false;
          const priority = cfg?.priority ?? 99;
          const hasTradesData = s.trades > 0;

          return (
            <div key={s.code} className={`strat-item ${enabled ? '' : 'disabled'}`}>
              <StrategyTag code={s.code} size={28} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 12, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
                  {s.name}
                  {enabled && (
                    <span style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 600 }}>P{priority}</span>
                  )}
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 1, lineHeight: 1.3 }}>
                  {s.description}
                </div>
              </div>
              {hasTradesData ? (
                <div style={{ textAlign: 'right', minWidth: 70 }}>
                  <div style={{ fontSize: 13, fontWeight: 800 }} className={s.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                    {s.pnl >= 0 ? '+' : ''}${s.pnl.toFixed(0)}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    {s.trades}T · {s.win_rate.toFixed(0)}%
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'right', minWidth: 70 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>—</div>
                </div>
              )}
              <span className={`cat-badge ${catClass(s.category)}`}>{s.category}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
