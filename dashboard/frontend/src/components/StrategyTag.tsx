const STRATEGY_COLORS: Record<string, string> = {
  H: '#e74c3c',
  G: '#e67e22',
  A: '#f1c40f',
  F: '#2ecc71',
  D: '#1abc9c',
  V: '#3498db',
  P: '#9b59b6',
  M: '#e91e63',
  R: '#ff5722',
  W: '#795548',
  O: '#00bcd4',
  B: '#4caf50',
  K: '#ff9800',
  C: '#673ab7',
  S: '#009688',
  E: '#f44336',
  I: '#2196f3',
  J: '#8bc34a',
  N: '#ff6f00',
  L: '#ffd700',
};

export function strategyColor(code: string): string {
  return STRATEGY_COLORS[code] || '#555';
}

export function StrategyTag({ code, size = 24 }: { code: string; size?: number }) {
  return (
    <span
      className="strat-tag"
      style={{
        width: size,
        height: size,
        fontSize: size * 0.45,
        background: `linear-gradient(135deg, ${strategyColor(code)}, ${strategyColor(code)}cc)`,
      }}
      title={code}
    >
      {code}
    </span>
  );
}
