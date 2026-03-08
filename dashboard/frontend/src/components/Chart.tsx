import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts';
import { api, ChartData } from '../api/client';
import { strategyColor } from './StrategyTag';

interface Props {
  symbol: string | null;
}

export function Chart({ symbol }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const [chartData, setChartData] = useState<ChartData | null>(null);

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
      },
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
    };
  }, []);

  useEffect(() => {
    if (!symbol) return;

    const fetchData = async () => {
      try {
        const data = await api.chart(symbol);
        setChartData(data);

        if (!candleRef.current || !volumeRef.current) return;

        if (data.bars.length === 0) {
          candleRef.current.setData([]);
          volumeRef.current.setData([]);
          return;
        }

        const candles: CandlestickData<Time>[] = data.bars.map((b) => ({
          time: (new Date(b.time).getTime() / 1000) as Time,
          open: b.open,
          high: b.high,
          low: b.low,
          close: b.close,
        }));

        const volumes = data.bars.map((b) => ({
          time: (new Date(b.time).getTime() / 1000) as Time,
          value: b.volume,
          color: b.close >= b.open ? 'rgba(0,230,118,0.15)' : 'rgba(255,82,82,0.15)',
        }));

        candleRef.current.setData(candles);
        volumeRef.current.setData(volumes);

        const markers = data.markers.map((m) => ({
          time: (new Date(m.time).getTime() / 1000) as Time,
          position: m.type === 'buy' ? ('belowBar' as const) : ('aboveBar' as const),
          color: m.type === 'buy' ? strategyColor(m.strategy) : (m.pnl && m.pnl >= 0 ? '#00e676' : '#ff5252'),
          shape: m.type === 'buy' ? ('arrowUp' as const) : ('arrowDown' as const),
          text: m.text,
        }));

        if (markers.length > 0) {
          candleRef.current.setMarkers(markers);
        }

        chartRef.current?.timeScale().fitContent();
      } catch (e) {
        console.error('Chart fetch error:', e);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [symbol]);

  return (
    <div className="card animate-in">
      <div className="chart-header">
        <div>
          <span className="chart-symbol">{symbol || 'Select a ticker'}</span>
          {chartData && chartData.bars.length > 0 && (
            <span className="chart-meta">{chartData.bars.length} candles · 2min</span>
          )}
        </div>
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
      <div ref={containerRef} style={{ height: 420, width: '100%' }} />
    </div>
  );
}
