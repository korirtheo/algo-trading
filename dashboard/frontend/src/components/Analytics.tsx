import { useState, useEffect } from 'react';
import { fetchJSON } from '../api/client';

type Tab = 'trades' | 'signals' | 'orders' | 'snapshots' | 'events' | 'bars' | 'watchlist' | 'slippage' | 'feed_comparison' | 'intraday_discoveries';

interface Signal {
  id: number;
  date: string;
  timestamp: string;
  ticker: string;
  strategy: string;
  signal_price: number;
  action: string;
  reason: string | null;
  order_id: string | null;
  gap_pct: number | null;
}

interface OrderEvent {
  id: number;
  timestamp: string;
  order_id: string;
  ticker: string;
  strategy: string;
  side: string;
  event_type: string;
  signal_price: number | null;
  fill_price: number | null;
  filled_qty: number | null;
  slip_bp: number | null;
  status: string | null;
}

interface AccountSnapshot {
  id: number;
  timestamp: string;
  snapshot_type: string;
  cash: number;
  equity: number;
  buying_power: number;
  portfolio_value: number;
  daily_pnl: number | null;
  trades_count: number | null;
  positions_count: number | null;
}

interface SystemEvent {
  id: number;
  timestamp: string;
  event_type: string;
  severity: string;
  message: string;
  details: string | null;
}

interface BarSummary {
  id: number;
  date: string;
  ticker: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap: number | null;
  bar_count: number;
}

interface WatchlistItem {
  id: number;
  date: string;
  ticker: string;
  gap_pct: number | null;
  pm_volume: number | null;
  float_shares: number | null;
  scan_time: string;
}

interface TradeDetail {
  id: number;
  ticker: string;
  strategy: string;
  entry_price: number;
  exit_price: number;
  shares: number;
  pnl: number;
  pnl_pct: number;
  reason: string;
  entry_time: string;
  exit_time: string;
  deployed_amount: number | null;
  stop_price: number | null;
  target_price: number | null;
  peak_price: number | null;
  trail_pct: number | null;
  time_limit_min: number | null;
  hold_time_min: number | null;
}

interface SlippageStats {
  date: string;
  n_fills: number;
  avg_slip_bp: number | null;
  median_slip_bp: number | null;
  max_slip_bp: number | null;
  min_slip_bp: number | null;
  p95_slip_bp: number | null;
  dollar_volume: number;
  realized_cost: number;
}

interface SlippageByStrategy {
  strategy: string;
  n: number;
  n_buys: number;
  n_sells: number;
  avg_slip_bp: number | null;
  median_slip_bp: number | null;
  min_slip_bp: number | null;
  max_slip_bp: number | null;
  avg_buy_slip_bp: number | null;
  avg_sell_slip_bp: number | null;
  dollar_volume: number;
  realized_cost: number;
}

interface SlippageRow {
  id: number;
  timestamp: string;
  order_id: string;
  ticker: string;
  strategy: string;
  side: string;
  event_type: string;
  signal_price: number | null;
  fill_price: number | null;
  filled_qty: number | null;
  slip_bp: number | null;
  status: string | null;
  dollar_amount: number;
  participation_rate: number | null;
}

interface FeedComparisonRow {
  bar_time: string;
  ticker: string;
  tradier_open: number | null;
  tradier_high: number | null;
  tradier_low: number | null;
  tradier_close: number | null;
  tradier_volume: number | null;
  alpaca_open: number | null;
  alpaca_high: number | null;
  alpaca_low: number | null;
  alpaca_close: number | null;
  alpaca_volume: number | null;
  close_diff_bp: number | null;
  vol_diff_pct: number | null;
}

interface IntradayDiscovery {
  id: number;
  timestamp: string;
  ticker: string;
  price: number;
  percent_change: number;
  source: string;
  gap_pct: number | null;
  cumulative_volume: number | null;
  volume: number | null;
  float_shares: number | null;
}

export const Analytics = () => {
  const [tab, setTab] = useState<Tab>('trades');
  const [date, setDate] = useState(() => {
    const today = new Date();
    return today.toISOString().split('T')[0];
  });

  const [tradeDetails, setTradeDetails] = useState<TradeDetail[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [orders, setOrders] = useState<OrderEvent[]>([]);
  const [snapshots, setSnapshots] = useState<AccountSnapshot[]>([]);
  const [events, setEvents] = useState<SystemEvent[]>([]);
  const [bars, setBars] = useState<BarSummary[]>([]);
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [slippageStats, setSlippageStats] = useState<SlippageStats | null>(null);
  const [slippageByStrategy, setSlippageByStrategy] = useState<SlippageByStrategy[]>([]);
  const [slippageRows, setSlippageRows] = useState<SlippageRow[]>([]);
  const [feedComparison, setFeedComparison] = useState<FeedComparisonRow[]>([]);
  const [intradayDiscoveries, setIntradayDiscoveries] = useState<IntradayDiscovery[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadData();
  }, [tab, date]);

  const loadData = async () => {
    setLoading(true);
    try {
      switch (tab) {
        case 'trades': {
          const tradesData = await fetchJSON<TradeDetail[]>(`/api/analytics/trades/details/${date}`);
          setTradeDetails(tradesData);
          break;
        }
        case 'signals': {
          const signalsData = await fetchJSON<Signal[]>(`/api/analytics/signals/${date}`);
          setSignals(signalsData);
          break;
        }
        case 'orders': {
          const ordersData = await fetchJSON<OrderEvent[]>(`/api/analytics/orders/${date}`);
          setOrders(ordersData);
          break;
        }
        case 'snapshots': {
          const snapshotsData = await fetchJSON<AccountSnapshot[]>(`/api/analytics/snapshots/${date}`);
          setSnapshots(snapshotsData);
          break;
        }
        case 'events': {
          const eventsData = await fetchJSON<SystemEvent[]>(`/api/analytics/events/${date}`);
          setEvents(eventsData);
          break;
        }
        case 'bars': {
          const barsData = await fetchJSON<BarSummary[]>(`/api/analytics/bars/${date}`);
          setBars(barsData);
          break;
        }
        case 'watchlist': {
          const watchlistData = await fetchJSON<WatchlistItem[]>(`/api/analytics/watchlist/${date}`);
          setWatchlist(watchlistData);
          break;
        }
        case 'slippage': {
          const slippageData = await fetchJSON<{ stats: SlippageStats; by_strategy: SlippageByStrategy[]; rows: SlippageRow[] }>(`/api/analytics/slippage/${date}`);
          setSlippageStats(slippageData.stats);
          setSlippageByStrategy(slippageData.by_strategy);
          setSlippageRows(slippageData.rows);
          break;
        }
        case 'feed_comparison': {
          const feedData = await fetchJSON<FeedComparisonRow[]>(`/api/analytics/feed_comparison/${date}`);
          setFeedComparison(feedData);
          break;
        }
        case 'intraday_discoveries': {
          const discoveriesData = await fetchJSON<IntradayDiscovery[]>(`/api/analytics/intraday_discoveries/${date}`);
          setIntradayDiscoveries(discoveriesData);
          break;
        }
      }
    } catch (err) {
      console.error('Failed to load analytics:', err);
    } finally {
      setLoading(false);
    }
  };

  const changeDate = (offset: number) => {
    const d = new Date(date);
    d.setDate(d.getDate() + offset);
    setDate(d.toISOString().split('T')[0]);
  };

  const renderTradeDetails = () => (
    <div className="analytics-table-container">
      <table className="analytics-table">
        <thead>
          <tr>
            <th>Entry Time</th>
            <th>Ticker</th>
            <th>Strat</th>
            <th>Entry $</th>
            <th>Exit $</th>
            <th>Peak $</th>
            <th>Stop $</th>
            <th>Target $</th>
            <th>Trail %</th>
            <th>Time Limit</th>
            <th>Hold Time</th>
            <th>Shares</th>
            <th>Deployed</th>
            <th>P&L</th>
            <th>P&L %</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {tradeDetails.map((t) => {
            const pnlClass = t.pnl > 0 ? 'positive' : t.pnl < 0 ? 'negative' : '';
            const peakAboveTarget = t.peak_price && t.target_price && t.peak_price >= t.target_price;
            const exitBelowStop = t.exit_price && t.stop_price && t.exit_price <= t.stop_price;
            return (
              <tr key={t.id}>
                <td>{new Date(t.entry_time).toLocaleTimeString()}</td>
                <td className="ticker-cell">{t.ticker}</td>
                <td>{t.strategy}</td>
                <td>${t.entry_price.toFixed(2)}</td>
                <td className={pnlClass}>${t.exit_price.toFixed(2)}</td>
                <td className={peakAboveTarget ? 'positive' : ''}>
                  {t.peak_price ? `$${t.peak_price.toFixed(2)}` : '-'}
                </td>
                <td className={exitBelowStop ? 'negative' : ''}>
                  {t.stop_price ? `$${t.stop_price.toFixed(2)}` : '-'}
                </td>
                <td>{t.target_price ? `$${t.target_price.toFixed(2)}` : '-'}</td>
                <td>{t.trail_pct !== null ? `${t.trail_pct.toFixed(1)}%` : '-'}</td>
                <td>{t.time_limit_min ? `${t.time_limit_min}m` : '-'}</td>
                <td>{t.hold_time_min ? `${t.hold_time_min.toFixed(1)}m` : '-'}</td>
                <td>{t.shares}</td>
                <td>{t.deployed_amount ? `$${t.deployed_amount.toFixed(0)}` : '-'}</td>
                <td className={pnlClass}>{t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}</td>
                <td className={pnlClass}>{t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct.toFixed(1)}%</td>
                <td style={{ fontSize: 11 }}>{t.reason}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {tradeDetails.length === 0 && !loading && <div className="empty-state">No trades for this date</div>}
    </div>
  );

  const renderSignals = () => (
    <div className="analytics-table-container">
      <table className="analytics-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Ticker</th>
            <th>Strategy</th>
            <th>Price</th>
            <th>Gap%</th>
            <th>Action</th>
            <th>Reason</th>
            <th>Order ID</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s) => (
            <tr key={s.id}>
              <td>{s.timestamp ? new Date(s.timestamp).toLocaleTimeString() : '—'}</td>
              <td className="ticker-cell">{s.ticker}</td>
              <td>{s.strategy}</td>
              <td>${s.signal_price.toFixed(2)}</td>
              <td className={s.gap_pct && s.gap_pct > 0 ? 'positive' : ''}>{s.gap_pct?.toFixed(1)}%</td>
              <td>
                <span className={`badge ${s.action === 'TAKEN' ? 'badge-success' : 'badge-warning'}`}>
                  {s.action}
                </span>
              </td>
              <td>{s.reason || '-'}</td>
              <td className="order-id-cell">{s.order_id || '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {signals.length === 0 && !loading && <div className="empty-state">No signals for this date</div>}
    </div>
  );

  const renderOrders = () => (
    <div className="analytics-table-container">
      <table className="analytics-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Order ID</th>
            <th>Ticker</th>
            <th>Strategy</th>
            <th>Side</th>
            <th>Event</th>
            <th>Signal $</th>
            <th>Fill $</th>
            <th>Qty</th>
            <th>Slip (bp)</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {orders.map((o) => (
            <tr key={o.id}>
              <td>{new Date(o.timestamp).toLocaleTimeString()}</td>
              <td className="order-id-cell">{o.order_id.substring(0, 8)}...</td>
              <td className="ticker-cell">{o.ticker}</td>
              <td>{o.strategy}</td>
              <td className={o.side === 'buy' ? 'positive' : 'negative'}>{o.side.toUpperCase()}</td>
              <td>{o.event_type}</td>
              <td>{o.signal_price ? `$${o.signal_price.toFixed(2)}` : '-'}</td>
              <td>{o.fill_price ? `$${o.fill_price.toFixed(2)}` : '-'}</td>
              <td>{o.filled_qty || '-'}</td>
              <td className={o.slip_bp && o.slip_bp > 0 ? 'negative' : o.slip_bp ? 'positive' : ''}>
                {o.slip_bp !== null ? o.slip_bp.toFixed(1) : '-'}
              </td>
              <td>{o.status || '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {orders.length === 0 && !loading && <div className="empty-state">No order events for this date</div>}
    </div>
  );

  const renderSnapshots = () => (
    <div className="analytics-table-container">
      <table className="analytics-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Type</th>
            <th>Equity</th>
            <th>Cash</th>
            <th>Buying Power</th>
            <th>Portfolio Value</th>
            <th>Daily P&L</th>
            <th>Trades</th>
            <th>Positions</th>
          </tr>
        </thead>
        <tbody>
          {snapshots.map((s) => (
            <tr key={s.id}>
              <td>{new Date(s.timestamp).toLocaleTimeString()}</td>
              <td>{s.snapshot_type}</td>
              <td className="positive">${s.equity.toLocaleString()}</td>
              <td>${s.cash.toLocaleString()}</td>
              <td>${s.buying_power.toLocaleString()}</td>
              <td>${s.portfolio_value.toLocaleString()}</td>
              <td className={s.daily_pnl && s.daily_pnl > 0 ? 'positive' : s.daily_pnl && s.daily_pnl < 0 ? 'negative' : ''}>
                {s.daily_pnl !== null ? `$${s.daily_pnl.toFixed(2)}` : '-'}
              </td>
              <td>{s.trades_count ?? '-'}</td>
              <td>{s.positions_count ?? '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {snapshots.length === 0 && !loading && <div className="empty-state">No snapshots for this date</div>}
    </div>
  );

  const renderEvents = () => (
    <div className="analytics-table-container">
      <table className="analytics-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Type</th>
            <th>Severity</th>
            <th>Message</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {[...events].sort((a, b) => {
            // Errors first, then warnings, then info
            const order: Record<string, number> = { critical: 0, error: 1, warning: 2, info: 3 };
            return (order[a.severity] ?? 4) - (order[b.severity] ?? 4);
          }).map((e) => (
            <tr key={e.id} style={e.severity === 'error' || e.severity === 'critical' ? { background: 'var(--red-bg)' } : undefined}>
              <td>{new Date(e.timestamp).toLocaleTimeString()}</td>
              <td>{e.event_type}</td>
              <td>
                <span className={`badge ${
                  e.severity === 'critical' ? 'badge-error' :
                  e.severity === 'error' ? 'badge-error' :
                  e.severity === 'warning' ? 'badge-warning' :
                  'badge-info'
                }`}>
                  {e.severity.toUpperCase()}
                </span>
              </td>
              <td style={(e.severity === 'error' || e.severity === 'critical') ? { fontWeight: 600 } : undefined}>{e.message}</td>
              <td className="details-cell">{e.details || '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {events.length === 0 && !loading && <div className="empty-state">No system events for this date</div>}
    </div>
  );

  const renderBars = () => (
    <div className="analytics-table-container">
      <table className="analytics-table">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Open</th>
            <th>High</th>
            <th>Low</th>
            <th>Close</th>
            <th>Volume</th>
            <th>VWAP</th>
            <th>Bars</th>
          </tr>
        </thead>
        <tbody>
          {bars.map((b) => (
            <tr key={b.id}>
              <td className="ticker-cell">{b.ticker}</td>
              <td>${b.open.toFixed(2)}</td>
              <td>${b.high.toFixed(2)}</td>
              <td>${b.low.toFixed(2)}</td>
              <td className={b.close > b.open ? 'positive' : b.close < b.open ? 'negative' : ''}>
                ${b.close.toFixed(2)}
              </td>
              <td>{b.volume.toLocaleString()}</td>
              <td>{b.vwap ? `$${b.vwap.toFixed(2)}` : '-'}</td>
              <td>{b.bar_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {bars.length === 0 && !loading && <div className="empty-state">No bar summaries for this date</div>}
    </div>
  );

  const renderWatchlist = () => (
    <div className="analytics-table-container">
      <table className="analytics-table">
        <thead>
          <tr>
            <th>Scan Time</th>
            <th>Ticker</th>
            <th>Gap%</th>
            <th>PM Volume</th>
            <th>Float</th>
          </tr>
        </thead>
        <tbody>
          {watchlist.map((w) => (
            <tr key={w.id}>
              <td>{new Date(w.scan_time).toLocaleTimeString()}</td>
              <td className="ticker-cell">{w.ticker}</td>
              <td className={w.gap_pct && w.gap_pct > 0 ? 'positive' : ''}>
                {w.gap_pct !== null ? `${w.gap_pct.toFixed(1)}%` : '-'}
              </td>
              <td>{w.pm_volume !== null ? w.pm_volume.toLocaleString() : '-'}</td>
              <td>{w.float_shares !== null ? `${(w.float_shares / 1e6).toFixed(1)}M` : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {watchlist.length === 0 && !loading && <div className="empty-state">No watchlist for this date</div>}
    </div>
  );

  const renderIntradayDiscoveries = () => (
    <div className="analytics-table-container">
      <table className="analytics-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Ticker</th>
            <th style={{ textAlign: 'right' }}>Gap%</th>
            <th style={{ textAlign: 'right' }}>Cumul. Volume</th>
            <th style={{ textAlign: 'right' }}>Volume</th>
            <th style={{ textAlign: 'right' }}>Float</th>
          </tr>
        </thead>
        <tbody>
          {intradayDiscoveries.map((d) => (
            <tr key={d.id}>
              <td>{new Date(d.timestamp).toLocaleTimeString()}</td>
              <td className="ticker-cell">{d.ticker}</td>
              <td style={{ textAlign: 'right' }}>
                {d.gap_pct != null ? <span className="positive">+{d.gap_pct.toFixed(1)}%</span> : '—'}
              </td>
              <td style={{ textAlign: 'right' }}>
                {d.cumulative_volume != null ? d.cumulative_volume.toLocaleString() : '—'}
              </td>
              <td style={{ textAlign: 'right' }}>
                {d.volume != null ? d.volume.toLocaleString() : '—'}
              </td>
              <td style={{ textAlign: 'right' }}>
                {d.float_shares != null ? `${(d.float_shares / 1e6).toFixed(1)}M` : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {intradayDiscoveries.length === 0 && !loading && <div className="empty-state">No intraday discoveries for this date</div>}
    </div>
  );

  const fmtBp = (v: number | null) => {
    if (v === null || v === undefined) return '—';
    return `${v >= 0 ? '+' : ''}${v.toFixed(1)} bp`;
  };

  const bpColor = (v: number | null) => {
    if (v === null || v === undefined) return 'var(--text-secondary)';
    if (v > 20) return 'var(--red)';
    if (v < -10) return 'var(--green)';
    return 'var(--text-secondary)';
  };

  const renderSlippage = () => (
    <div>
      {slippageStats && (
        <>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
            gap: 12,
            padding: '12px',
            borderBottom: '1px solid var(--border)',
            fontSize: 12,
          }}>
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>Fills</div>
              <div style={{ fontWeight: 600 }}>{slippageStats.n_fills}</div>
            </div>
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>Avg Slip</div>
              <div style={{ fontWeight: 600, color: bpColor(slippageStats.avg_slip_bp) }}>
                {fmtBp(slippageStats.avg_slip_bp)}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>Median</div>
              <div style={{ fontWeight: 600, color: bpColor(slippageStats.median_slip_bp) }}>
                {fmtBp(slippageStats.median_slip_bp)}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>P95</div>
              <div style={{ fontWeight: 600, color: bpColor(slippageStats.p95_slip_bp) }}>
                {fmtBp(slippageStats.p95_slip_bp)}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>Best</div>
              <div style={{ fontWeight: 600, color: bpColor(slippageStats.min_slip_bp) }}>
                {fmtBp(slippageStats.min_slip_bp)}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>Worst</div>
              <div style={{ fontWeight: 600, color: bpColor(slippageStats.max_slip_bp) }}>
                {fmtBp(slippageStats.max_slip_bp)}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>$ Volume</div>
              <div style={{ fontWeight: 600 }}>
                ${slippageStats.dollar_volume.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>Slip Cost</div>
              <div style={{
                fontWeight: 600,
                color: slippageStats.realized_cost > 0 ? 'var(--red)' : 'var(--green)',
              }}>
                ${slippageStats.realized_cost.toFixed(2)}
              </div>
            </div>
          </div>

          {slippageByStrategy.length > 0 && (
            <div style={{ borderBottom: '1px solid var(--border)' }}>
              <div style={{
                fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase',
                letterSpacing: 0.5, padding: '8px 12px 4px',
              }}>
                By Strategy
              </div>
              <table className="analytics-table" style={{ marginBottom: 4 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left' }}>Strategy</th>
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
                  {slippageByStrategy.map((s) => (
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

          {slippageRows.length > 0 && (
            <div className="analytics-table-container">
              <table className="analytics-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Ticker</th>
                    <th>Strat</th>
                    <th>Side</th>
                    <th style={{ textAlign: 'right' }}>Signal</th>
                    <th style={{ textAlign: 'right' }}>Fill</th>
                    <th style={{ textAlign: 'right' }}>Slip BP</th>
                    <th style={{ textAlign: 'right' }}>Qty</th>
                    <th style={{ textAlign: 'right' }}>Particip.</th>
                  </tr>
                </thead>
                <tbody>
                  {slippageRows.map((r) => (
                    <tr key={r.id}>
                      <td>{new Date(r.timestamp).toLocaleTimeString()}</td>
                      <td className="ticker-cell">{r.ticker}</td>
                      <td>{r.strategy}</td>
                      <td>{r.side.toUpperCase()}</td>
                      <td style={{ textAlign: 'right' }}>
                        ${r.signal_price ? r.signal_price.toFixed(2) : '-'}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        ${r.fill_price ? r.fill_price.toFixed(2) : '-'}
                      </td>
                      <td style={{
                        textAlign: 'right',
                        fontWeight: 600,
                        color: bpColor(r.slip_bp),
                      }}>
                        {fmtBp(r.slip_bp)}
                      </td>
                      <td style={{ textAlign: 'right' }}>{r.filled_qty}</td>
                      <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>
                        {r.participation_rate != null ? `${(r.participation_rate * 100).toFixed(3)}%` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
      {!slippageStats && !loading && <div className="empty-state">No slippage data for this date</div>}
    </div>
  );

  const renderFeedComparison = () => {
    const fmtBp = (v: number | null) => v !== null ? `${v > 0 ? '+' : ''}${v.toFixed(1)} bp` : '-';
    const fmtPct = (v: number | null) => v !== null ? `${v > 0 ? '+' : ''}${v.toFixed(1)}%` : '-';
    const bpColor = (v: number | null) => {
      if (v === null) return undefined;
      const abs = Math.abs(v);
      if (abs > 50) return 'var(--red)';
      if (abs > 20) return '#f59e0b';
      return 'var(--green)';
    };

    // Summary stats
    const rows = feedComparison;
    const bothPresent = rows.filter(r => r.tradier_close !== null && r.alpaca_close !== null);
    const tradierOnly = rows.filter(r => r.tradier_close !== null && r.alpaca_close === null);
    const alpacaOnly = rows.filter(r => r.tradier_close === null && r.alpaca_close !== null);
    const diffs = bothPresent.map(r => r.close_diff_bp).filter((v): v is number => v !== null);
    const avgDiff = diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : null;
    const absDiffs = diffs.map(Math.abs);
    const avgAbsDiff = absDiffs.length ? absDiffs.reduce((a, b) => a + b, 0) / absDiffs.length : null;

    return (
      <div className="analytics-table-container">
        {rows.length > 0 && (
          <div style={{
            display: 'flex', gap: 24, padding: '10px 12px', flexWrap: 'wrap',
            borderBottom: '1px solid var(--border)', fontSize: 12, marginBottom: 8,
          }}>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Total Bars </span>
              <strong>{rows.length}</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Both Sources </span>
              <strong>{bothPresent.length}</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Tradier Only </span>
              <strong style={{ color: tradierOnly.length > 0 ? '#f59e0b' : undefined }}>{tradierOnly.length}</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Alpaca Only </span>
              <strong style={{ color: alpacaOnly.length > 0 ? '#f59e0b' : undefined }}>{alpacaOnly.length}</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Avg Close Diff </span>
              <strong style={{ color: bpColor(avgDiff) }}>{fmtBp(avgDiff !== null ? Math.round(avgDiff * 10) / 10 : null)}</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Avg |Diff| </span>
              <strong style={{ color: bpColor(avgAbsDiff !== null ? avgAbsDiff : null) }}>{fmtBp(avgAbsDiff !== null ? Math.round(avgAbsDiff * 10) / 10 : null)}</strong>
            </div>
          </div>
        )}
        <table className="analytics-table">
          <thead>
            <tr>
              <th>Bar Time</th>
              <th>Ticker</th>
              <th style={{ textAlign: 'right' }}>Tradier Close</th>
              <th style={{ textAlign: 'right' }}>Tradier Vol</th>
              <th style={{ textAlign: 'right' }}>Alpaca Close</th>
              <th style={{ textAlign: 'right' }}>Alpaca Vol</th>
              <th style={{ textAlign: 'right' }}>Close Δ (bp)</th>
              <th style={{ textAlign: 'right' }}>Vol Δ (%)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>{r.bar_time}</td>
                <td className="ticker-cell">{r.ticker}</td>
                <td style={{ textAlign: 'right' }}>
                  {r.tradier_close !== null ? `$${r.tradier_close.toFixed(2)}` : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                </td>
                <td style={{ textAlign: 'right' }}>
                  {r.tradier_volume !== null ? r.tradier_volume.toLocaleString() : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                </td>
                <td style={{ textAlign: 'right' }}>
                  {r.alpaca_close !== null ? `$${r.alpaca_close.toFixed(2)}` : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                </td>
                <td style={{ textAlign: 'right' }}>
                  {r.alpaca_volume !== null ? r.alpaca_volume.toLocaleString() : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                </td>
                <td style={{ textAlign: 'right', fontWeight: 600, color: bpColor(r.close_diff_bp) }}>
                  {fmtBp(r.close_diff_bp)}
                </td>
                <td style={{ textAlign: 'right', color: r.vol_diff_pct !== null && Math.abs(r.vol_diff_pct) > 20 ? '#f59e0b' : undefined }}>
                  {fmtPct(r.vol_diff_pct)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && !loading && (
          <div className="empty-state">No feed comparison data for this date</div>
        )}
      </div>
    );
  };

  return (
    <div className="analytics-page">
      <div className="analytics-header">
        <div className="analytics-title">
          <h2>Analytics & Database</h2>
        </div>
        <div className="date-nav">
          <button onClick={() => changeDate(-1)} className="nav-btn">←</button>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="date-input"
          />
          <button onClick={() => changeDate(1)} className="nav-btn">→</button>
        </div>
      </div>

      <div className="analytics-tabs">
        <button
          className={`tab-btn ${tab === 'trades' ? 'active' : ''}`}
          onClick={() => setTab('trades')}
        >
          Trade Details
        </button>
        <button
          className={`tab-btn ${tab === 'signals' ? 'active' : ''}`}
          onClick={() => setTab('signals')}
        >
          Signals
        </button>
        <button
          className={`tab-btn ${tab === 'orders' ? 'active' : ''}`}
          onClick={() => setTab('orders')}
        >
          Orders
        </button>
        <button
          className={`tab-btn ${tab === 'snapshots' ? 'active' : ''}`}
          onClick={() => setTab('snapshots')}
        >
          Snapshots
        </button>
        <button
          className={`tab-btn ${tab === 'events' ? 'active' : ''}`}
          onClick={() => setTab('events')}
        >
          System Events
        </button>
        <button
          className={`tab-btn ${tab === 'bars' ? 'active' : ''}`}
          onClick={() => setTab('bars')}
        >
          Bar Summaries
        </button>
        <button
          className={`tab-btn ${tab === 'watchlist' ? 'active' : ''}`}
          onClick={() => setTab('watchlist')}
        >
          Watchlist
        </button>
        <button
          className={`tab-btn ${tab === 'slippage' ? 'active' : ''}`}
          onClick={() => setTab('slippage')}
        >
          Slippage
        </button>
        <button
          className={`tab-btn ${tab === 'feed_comparison' ? 'active' : ''}`}
          onClick={() => setTab('feed_comparison')}
        >
          Feed Comparison
        </button>
        <button
          className={`tab-btn ${tab === 'intraday_discoveries' ? 'active' : ''}`}
          onClick={() => setTab('intraday_discoveries')}
        >
          Intraday Discoveries
        </button>
      </div>

      <div className="analytics-content">
        {loading ? (
          <div className="loading-state">Loading...</div>
        ) : (
          <>
            {tab === 'trades' && renderTradeDetails()}
            {tab === 'signals' && renderSignals()}
            {tab === 'orders' && renderOrders()}
            {tab === 'snapshots' && renderSnapshots()}
            {tab === 'events' && renderEvents()}
            {tab === 'bars' && renderBars()}
            {tab === 'watchlist' && renderWatchlist()}
            {tab === 'slippage' && renderSlippage()}
            {tab === 'feed_comparison' && renderFeedComparison()}
            {tab === 'intraday_discoveries' && renderIntradayDiscoveries()}
          </>
        )}
      </div>
    </div>
  );
};
