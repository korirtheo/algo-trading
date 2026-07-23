import { useState, useEffect } from 'react';
import { api, HealthCheck } from '../api/client';

export function SystemHealth() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [checks, setChecks] = useState<HealthCheck[]>([]);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await api.health();
        setHealthy(data.healthy);
        setChecks(data.checks);
      } catch {
        setHealthy(null);
        setChecks([]);
      }
    };
    fetchHealth();
    const id = setInterval(fetchHealth, 10000);
    return () => clearInterval(id);
  }, []);

  // No data yet
  if (healthy === null && checks.length === 0) return null;

  const errors = checks.filter(c => c.status === 'error');
  const warnings = checks.filter(c => c.status === 'warning');
  const hasIssues = errors.length > 0 || warnings.length > 0;

  if (!hasIssues) {
    return (
      <div className="health-banner health-ok" onClick={() => setExpanded(!expanded)}>
        <span className="health-icon">&#10003;</span>
        <span className="health-text">All Systems Operational</span>
      </div>
    );
  }

  return (
    <div className={`health-banner ${errors.length > 0 ? 'health-error' : 'health-warning'}`} onClick={() => setExpanded(!expanded)}>
      <div className="health-summary">
        <span className="health-icon">{errors.length > 0 ? '&#9888;' : '&#9888;'}</span>
        <span className="health-text">
          {errors.length > 0
            ? `${errors.length} system error${errors.length > 1 ? 's' : ''}`
            : `${warnings.length} warning${warnings.length > 1 ? 's' : ''}`}
        </span>
        <span className="health-expand">{expanded ? '&#9650;' : '&#9660;'}</span>
      </div>
      {expanded && (
        <div className="health-details">
          {checks.map((c, i) => (
            <div key={i} className={`health-check health-check-${c.status}`}>
              <span className="health-check-status">
                {c.status === 'ok' ? '&#10003;' : c.status === 'warning' ? '&#9888;' : '&#10007;'}
              </span>
              <span className="health-check-component">{c.component}</span>
              <span className="health-check-message">{c.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
