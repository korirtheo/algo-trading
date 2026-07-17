import { useState } from 'react';
import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';
import { StrategyTag } from './StrategyTag';

function reasonClass(reason: string): string {
  if (reason === 'TARGET' || reason === 'TRAIL') return 'reason-win';
  if (reason === 'STOP') return 'reason-loss';
  return 'reason-neutral';
}

function formatTime(timeStr: string): string {
  if (!timeStr) return '';
  try {
    const dt = new Date(timeStr);
    return dt.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  } catch {
    return timeStr;
  }
}

export function TradeLog() {
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [historicalData, setHistoricalData] = useState<any>(null);

  const { data: trades } = usePolling(api.trades, 5000);

  // Show historical trades if a date is selected, otherwise show today's trades
  const displayTrades = selectedDate && historicalData ? historicalData.trades : trades;
  const wins = displayTrades?.filter((t: any) => t.pnl > 0).length || 0;
  const losses = displayTrades?.filter((t: any) => t.pnl <= 0).length || 0;
  const totalPnL = displayTrades?.reduce((sum: number, t: any) => sum + (t.pnl || 0), 0) || 0;

  const handleDateNav = async (offset: number) => {
    const today = new Date();
    const target = selectedDate ? new Date(selectedDate) : new Date(today);
    target.setDate(target.getDate() + offset);

    const dateStr = target.toISOString().split('T')[0];
    setSelectedDate(dateStr);

    try {
      const data = await api.tradesByDate(dateStr);
      setHistoricalData(data);
    } catch (error) {
      console.error('Failed to load trades for', dateStr, error);
      setHistoricalData({ trades: [], date: dateStr, found: false });
    }
  };

  const resetToToday = () => {
    setSelectedDate(null);
    setHistoricalData(null);
  };

  return (
    <div className="card animate-in">
      <div className="card-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
          <div>
            <span className="card-title">Trade Log</span>
            <span className="card-subtitle">
              {displayTrades?.length || 0} trades | <span style={{ color: '#4ade80', fontWeight: 700 }}>{wins}W</span> / <span style={{ color: '#f87171', fontWeight: 700 }}>{losses}L</span> |
              <span style={{ fontWeight: 700, color: totalPnL >= 0 ? '#4ade80' : '#f87171' }}> ${totalPnL.toFixed(0)}</span>
            </span>
          </div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            {selectedDate && (
              <button onClick={resetToToday} style={{ padding: '4px 8px', fontSize: '12px', cursor: 'pointer' }}>
                Today
              </button>
            )}
            <button onClick={() => handleDateNav(-1)} style={{ padding: '4px 8px', cursor: 'pointer' }}>
              ←
            </button>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              {selectedDate || 'Today'}
            </span>
            <button onClick={() => handleDateNav(1)} style={{ padding: '4px 8px', cursor: 'pointer' }}>
              →
            </button>
          </div>
        </div>
      </div>
      {(!displayTrades || displayTrades.length === 0) ? (
        <div className="empty-state">{selectedDate && historicalData?.found === false ? 'No trades found for this date' : 'No trades yet today'}</div>
      ) : (
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Ticker</th>
                <th style={{ textAlign: 'center' }}>Strat</th>
                <th style={{ textAlign: 'right' }}>Entry</th>
                <th style={{ textAlign: 'right' }}>Exit</th>
                <th style={{ textAlign: 'right' }}>Deployed</th>
                <th style={{ textAlign: 'right' }}>P&L</th>
                <th style={{ textAlign: 'right' }}>%</th>
                <th style={{ textAlign: 'center' }}>Entry Time</th>
                <th style={{ textAlign: 'center' }}>Exit Time</th>
                <th style={{ textAlign: 'center' }}>Reason</th>
              </tr>
            </thead>
            <tbody>
              {displayTrades.map((t: any, i: number) => {
                const cls = t.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                return (
                  <tr key={i}>
                    <td className="ticker-link">{t.ticker}</td>
                    <td style={{ textAlign: 'center' }}><StrategyTag code={t.strategy} size={20} /></td>
                    <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>${t.entry_price.toFixed(2)}</td>
                    <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>${t.exit_price.toFixed(2)}</td>
                    <td style={{ textAlign: 'right', color: 'var(--text-secondary)', fontSize: '11px' }}>
                      ${t.deployed_amount?.toFixed(0) || '—'}
                    </td>
                    <td style={{ textAlign: 'right', fontWeight: 700 }} className={cls}>
                      {t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(0)}
                    </td>
                    <td style={{ textAlign: 'right', fontWeight: 700 }} className={cls}>
                      {t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct.toFixed(1)}%
                    </td>
                    <td style={{ textAlign: 'center', fontSize: '11px', color: 'var(--text-secondary)' }}>
                      {formatTime(t.entry_time)}
                    </td>
                    <td style={{ textAlign: 'center', fontSize: '11px', color: 'var(--text-secondary)' }}>
                      {formatTime(t.exit_time)}
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <span className={`reason-badge ${reasonClass(t.reason)}`}>{t.reason}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
