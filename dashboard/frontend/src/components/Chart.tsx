import { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts';
import { api, ChartData } from '../api/client';
import { strategyColor } from './StrategyTag';
import { useWebSocket } from '../hooks/useWebSocket';

interface Props {
  symbol: string | null;
}

function toUnixTime(t: string): Time {
  return (new Date(t).getTime() / 1000) as Time;
}

export function Chart({ symbol }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const hasInitialFit = useRef(false);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const { lastMessage } = useWebSocket('/ws/live');

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: '#111620' },
        textColor: '#7a8699',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.03)' },
        horzLines: { color: 'rgba(255,255,255,0.03)' },
      },
      crosshair: {
        mode: 0,
        vertLine: { color: '#448aff', width: 1, style: 2 },
        horzLine: { color: '#448aff', width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.05)',
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.05)',
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
        tickMarkFormatter: (time: number) => {
          const d = new Date(time * 1000);
          const et = new Date(d.toLocaleString('en-US', { timeZone: 'America/New_York' }));
          const h = et.getHours().toString().padStart(2, '0');
          const m = et.getMinutes().toString().padStart(2, '0');
          return `${h}:${m}`;
        },
      },
      localization: {
        timeFormatter: (time: number) => {
          const d = new Date(time * 1000);
          return d.toLocaleString('en-US', {
            timeZone: 'America/New_York',
            month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit', hour12: false,
          });
        },
      },
      handleScroll: true,
      handleScale: true,
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00e676',
      downColor: '#ff5252',
      borderUpColor: '#00c853',
      borderDownColor: '#ff1744',
      wickUpColor: '#00e67688',
      wickDownColor: '#ff525288',
    });

    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });

    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
    });

    chartRef.current = chart;
    candleRef.current = candleSeries;
    volumeRef.current = volumeSeries;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      volumeRef.current = null;
    };
  }, []);

  // Load historical data when symbol changes
  const loadData = useCallback(async (sym: string) => {
    if (!candleRef.current || !volumeRef.current) return;
    try {
      const data = await api.chart(sym);
      setChartData(data);
      if (data.bars.length === 0) {
        candleRef.current.setData([]);
        volumeRef.current.setData([]);
        return;
      }

      const candles: CandlestickData<Time>[] = data.bars.map((b) => ({
        time: toUnixTime(b.time),
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      }));

      const volumes = data.bars.map((b) => ({
        time: toUnixTime(b.time),
        value: b.volume,
        color: b.close >= b.open ? 'rgba(0,230,118,0.18)' : 'rgba(255,82,82,0.18)',
      }));

      candleRef.current.setData(candles);
      volumeRef.current.setData(volumes);

      const markers = data.markers.map((m) => ({
        time: toUnixTime(m.time),
        position: m.type === 'buy' ? ('belowBar' as const) : ('aboveBar' as const),
        color: m.type === 'buy' ? strategyColor(m.strategy) : (m.pnl && m.pnl >= 0 ? '#00e676' : '#ff5252'),
        shape: m.type === 'buy' ? ('arrowUp' as const) : ('arrowDown' as const),
        text: m.text,
      }));
      if (markers.length > 0) candleRef.current.setMarkers(markers);

      // Only fit on first load — don't reset user's zoom on refresh
      if (!hasInitialFit.current) {
        chartRef.current?.timeScale().fitContent();
        hasInitialFit.current = true;
      }
    } catch (e) {
      console.error('Chart fetch error:', e);
    }
  }, []);

  // Reset and reload when symbol changes
  useEffect(() => {
    if (!symbol) return;
    hasInitialFit.current = false;
    candleRef.current?.setData([]);
    volumeRef.current?.setData([]);
    loadData(symbol);

    // Refresh markers every 15s but don't reset zoom
    const interval = setInterval(() => loadData(symbol), 15000);
    return () => clearInterval(interval);
  }, [symbol, loadData]);

  // Append new bars from WebSocket in real time
  useEffect(() => {
    if (!lastMessage || lastMessage.type !== 'bar') return;
    if (!symbol || lastMessage.symbol !== symbol) return;
    if (!candleRef.current || !volumeRef.current) return;

    const b = lastMessage.data;
    const t = toUnixTime(b.timestamp || b.time);
    candleRef.current.update({ time: t, open: b.open, high: b.high, low: b.low, close: b.close });
    volumeRef.current.update({
      time: t,
      value: b.volume,
      color: b.close >= b.open ? 'rgba(0,230,118,0.18)' : 'rgba(255,82,82,0.18)',
    });
  }, [lastMessage, symbol]);

  return (
    <div className="card animate-in" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="chart-header">
        <div>
          <span className="chart-symbol">{symbol || 'Select a ticker'}</span>
          {chartData && chartData.bars.length > 0 && (
            <span className="chart-meta">{chartData.bars.length} candles · 2min</span>
          )}
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          {/* Fit button to reset zoom */}
          {symbol && (
            <button
              className="chart-fit-btn"
              onClick={() => {
                chartRef.current?.timeScale().fitContent();
              }}
              title="Fit all candles"
            >
              ⤢ Fit
            </button>
          )}
          {chartData && chartData.markers.length > 0 && (
            <div className="chart-markers">
              {chartData.markers.map((m, i) => (
                <span
                  key={i}
                  className="chart-marker-badge"
                  style={{
                    background: m.type === 'buy' ? 'var(--blue-bg)' : (m.pnl && m.pnl >= 0 ? 'var(--green-bg)' : 'var(--red-bg)'),
                    color: m.type === 'buy' ? 'var(--blue)' : (m.pnl && m.pnl >= 0 ? 'var(--green)' : 'var(--red)'),
                    border: `1px solid ${m.type === 'buy' ? 'rgba(68,138,255,0.12)' : (m.pnl && m.pnl >= 0 ? 'rgba(0,230,118,0.12)' : 'rgba(255,82,82,0.12)')}`,
                  }}
                >
                  {m.text}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
      <div ref={containerRef} style={{ flex: 1, minHeight: 420, width: '100%' }} />
    </div>
  );
}
