import React from 'react';

export default function Footer() {
  return (
    <footer className="footer">
      <span>LightMamba-ASL &copy; {new Date().getFullYear()} — Academic Research Project</span>
      <div className="footer-right">
        <span>MobileNetV3 · MediaPipe · HMS-Mamba</span>
        <span style={{ color: 'var(--border-active)' }}>|</span>
        <span>Flask API · React CRA</span>
      </div>
    </footer>
  );
}
