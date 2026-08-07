import React from 'react';

export default function MetricCard({ icon, value, label, sub, accentColor = 'var(--accent-blue)', bgColor }) {
  return (
    <div className="stat-card">
      <div className="stat-icon" style={{ background: bgColor || `${accentColor}18`, color: accentColor }}>
        {icon}
      </div>
      <div className="stat-value" style={{ color: accentColor }}>{value}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}
