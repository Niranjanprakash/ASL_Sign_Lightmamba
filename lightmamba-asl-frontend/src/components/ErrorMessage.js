import React from 'react';

const AlertIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
);

export default function ErrorMessage({ message, onDismiss }) {
  if (!message) return null;
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: '10px',
      padding: '12px 16px', borderRadius: 'var(--radius-md)',
      background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
      color: '#fca5a5', fontSize: '0.85rem', lineHeight: '1.5',
    }}>
      <span style={{ flexShrink: 0, marginTop: '1px', color: 'var(--accent-red)' }}><AlertIcon /></span>
      <span style={{ flex: 1 }}>{message}</span>
      {onDismiss && (
        <button onClick={onDismiss} style={{
          background: 'none', border: 'none', color: '#fca5a5',
          cursor: 'pointer', padding: '0 2px', fontSize: '1rem', lineHeight: 1,
        }}>×</button>
      )}
    </div>
  );
}
