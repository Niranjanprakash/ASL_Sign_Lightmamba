import React from 'react';

export default function PageHeader({ label, title, subtitle, children }) {
  return (
    <div style={{ marginBottom: '28px' }}>
      {label && <div className="section-label">{label}</div>}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap' }}>
        <div>
          <h1 className="section-title">{title}</h1>
          {subtitle && <p className="section-subtitle">{subtitle}</p>}
        </div>
        {children && <div style={{ display: 'flex', gap: '10px', flexShrink: 0 }}>{children}</div>}
      </div>
    </div>
  );
}
