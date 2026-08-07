import React from 'react';

export default function StatusBadge({ status, label }) {
  const map = {
    online:    'badge-online',
    offline:   'badge-offline',
    warning:   'badge-warning',
    info:      'badge-info',
    active:    'badge-online',
    inactive:  'badge-offline',
    recording: 'badge-warning',
  };
  const cls = map[status] || 'badge-info';
  return <span className={`badge ${cls}`}>{label || status}</span>;
}
