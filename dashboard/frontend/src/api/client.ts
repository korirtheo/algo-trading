const BASE = '';

export async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export interface Account {
  cash: number;
  buying_power: number;
  equity: number;
  portfolio_value: number;
  daily_pnl: number;
  status: string;
  pdt: boolean;
  daytrade_count: number;
}

export interface Position {
  ticker: string;
  qty: number;
  avg_entry: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  strategy: string;
  entry_time: string;
  gap_pct: number;
}

export interface StrategyStatus {
  code: string;
  status: 'active' | 'done' | 'fired';
}

export interface WatchlistItem {
  ticker: string;
  gap_pct: number;
  pm_volume: number;
  premarket_high: number;
  prev_close: number;
  float_shares: number | null;
  status: string;
  strategy: string;
  eligible_strategies: StrategyStatus[];
  candle_count: number;
  last_price: number | null;
  change_pct: number | null;
}

export interface DiagnosticsStrategy {
  code: string;
  status: 'fired' | 'timed_out' | 'watching' | 'not_eligible';
}

export interface DiagnosticsItem {
  ticker: string;
  gap_pct: number;
  candle_count: number;
  premarket_high: number;
  market_open: number;
  pm_high_pct_above_open: number;
  last_price: number | null;
  change_pct: number | null;
  strategies: DiagnosticsStrategy[];
  traded: boolean;
  active_strategy: string;
  done: boolean;
}

export interface Trade {
  ticker: string;
  strategy: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  pnl_pct: number;
  reason: string;
  entry_time: string;
  exit_time: string;
}

export interface StrategyInfo {
  code: string;
  name: string;
  description: string;
  category: string;
  trades: number;
  wins: number;
  pnl: number;
  win_rate: number;
  best: number;
  worst: number;
}

export interface StrategyConfig {
  code: string;
  name: string;
  enabled: boolean;
  priority: number;
}

export interface ChartBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ChartMarker {
  time: string;
  type: 'buy' | 'sell';
  price: number;
  strategy: string;
  text: string;
  pnl?: number;
  reason?: string;
}

export interface ChartData {
  symbol: string;
  bars: ChartBar[];
  markers: ChartMarker[];
}

export interface Summary {
  active_position: string | null;
  daily_pnl: number;
  trades_count: number;
  wins: number;
  losses: number;
  candidates_count: number;
  tracking_count: number;
  active_feed?: string;
  feed_fallback?: boolean;
}

export interface HaltEvent {
  ticker: string;
  reason: string;
  halt_price: number | null;
  resume_price: number | null;
  halt_time: string | null;
  resume_ts: string | null;
  action: 'TRADED' | 'OPEN' | 'WATCHING' | 'SKIPPED';
  entry_price?: number | null;
  exit_price?: number | null;
  pnl?: number | null;
  reason_exit?: string;
}

export const api = {
  account: () => fetchJSON<Account>('/api/account'),
  positions: () => fetchJSON<Position[]>('/api/positions'),
  watchlist: () => fetchJSON<WatchlistItem[]>('/api/watchlist'),
  trades: () => fetchJSON<Trade[]>('/api/trades/today'),
  tradesByDate: (date: string) => fetchJSON<{trades: Trade[], date: string, found: boolean}>(`/api/trades/${date}`),
  strategies: () => fetchJSON<StrategyInfo[]>('/api/strategies'),
  strategyConfig: () => fetchJSON<StrategyConfig[]>('/api/strategies/config'),
  chart: (symbol: string) => fetchJSON<ChartData>(`/api/charts/${symbol}`),
  summary: () => fetchJSON<Summary>('/api/summary'),
  diagnostics: () => fetchJSON<DiagnosticsItem[]>('/api/diagnostics'),
  halts: () => fetchJSON<HaltEvent[]>('/api/halts/today'),
  slippage: () => fetchJSON<SlippageData>('/api/slippage/recent'),
  health: () => fetchJSON<{healthy: boolean | null, checks: HealthCheck[]}>('/api/health'),
};

export interface HealthCheck {
  component: string;
  status: 'ok' | 'warning' | 'error';
  message: string;
}

export interface SlippageRow {
  ts_signal: string;
  ts_fill: string;
  ticker: string;
  side: 'buy' | 'sell';
  strategy: string;
  signal_price: number | null;
  fill_price: number | null;
  slip_bp: number | null;
  qty: number | null;
  dollar_amount: number | null;
  cum_dollar_vol: number | null;
  participation_rate: number | null;
  status: string;
  order_id: string;
}

export interface SlippageStats {
  n_total: number;
  n_filled: number;
  avg_slip_bp: number | null;
  median_slip_bp: number | null;
  max_slip_bp: number | null;
  min_slip_bp: number | null;
  p95_slip_bp: number | null;
  dollar_volume_traded: number;
  realized_slip_cost: number;
}

export interface SlippageByStrategy {
  strategy: string;
  n: number;
  n_buys: number;
  n_sells: number;
  avg_slip_bp: number;
  median_slip_bp: number;
  min_slip_bp: number;
  max_slip_bp: number;
  avg_buy_slip_bp: number | null;
  avg_sell_slip_bp: number | null;
  dollar_volume: number;
  realized_cost: number;
}

export interface SlippageData {
  stats: SlippageStats;
  by_strategy: SlippageByStrategy[];
  rows: SlippageRow[];
}
