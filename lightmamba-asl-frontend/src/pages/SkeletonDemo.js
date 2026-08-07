import React from 'react';
import PageHeader from '../components/PageHeader';
import WebcamSkeleton from '../components/WebcamSkeleton';

export default function SkeletonDemo() {
  return (
    <div>
      <PageHeader
        label="MediaPipe Visualization"
        title="Skeletal Tracking"
        subtitle="Visualizing hand and upper-body landmarks used for temporal ASL recognition."
      />

      <div className="skeleton-layout">
        {/* ── LEFT: Webcam + Skeleton ─────────────────── */}
        <div>
          <WebcamSkeleton showControls={true} />
        </div>

        {/* ── RIGHT: Explanation ──────────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

          {/* Why skeleton */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div className="section-label" style={{ marginBottom: '10px' }}>Research Motivation</div>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '10px' }}>
              Why Skeleton Features?
            </h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
              RGB frames capture appearance, while MediaPipe landmarks provide an explicit geometric
              representation of hand and body movement. Tracking these landmarks over time allows
              LightMamba-ASL to model hand trajectories, finger configurations, two-hand coordination,
              and upper-body motion.
            </p>
          </div>

          {/* Landmark breakdown */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div className="section-label" style={{ marginBottom: '10px' }}>Landmark Breakdown</div>
            {[
              { label: 'Left Hand',  count: 21, color: 'var(--accent-blue)',   desc: 'Wrist + 5 fingers × 4 joints' },
              { label: 'Right Hand', count: 21, color: 'var(--accent-purple)', desc: 'Wrist + 5 fingers × 4 joints' },
              { label: 'Pose',       count: 33, color: 'var(--accent-cyan)',   desc: 'Upper-body joints + torso' },
            ].map(({ label, count, color, desc }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid var(--border-card)' }}>
                <div>
                  <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)' }}>{label}</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{desc}</div>
                </div>
                <div style={{ padding: '3px 10px', borderRadius: '12px', background: `${color}15`, border: `1px solid ${color}30`, fontSize: '0.78rem', fontWeight: 700, color }}>
                  {count} pts
                </div>
              </div>
            ))}
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingTop: '10px', fontSize: '0.8rem' }}>
              <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Total (all detected)</span>
              <span style={{ fontWeight: 800, color: 'var(--text-primary)' }}>75 landmarks</span>
            </div>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '8px' }}>
              Each landmark carries x, y, z coordinates. Missing landmarks are zero-filled with a validity mask.
            </p>
          </div>

          {/* Motion encoding */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div className="section-label" style={{ marginBottom: '10px' }}>Motion Encoding</div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px', padding: '8px 0' }}>
              {[
                { label: 'Frame t−1 Landmarks', color: 'var(--accent-purple)' },
                { label: 'Δt = Lt − Lt−1', color: 'var(--accent-amber)', isFormula: true },
                { label: 'Frame t Landmarks', color: 'var(--accent-blue)' },
              ].map(({ label, color, isFormula }, i) => (
                <React.Fragment key={label}>
                  {i > 0 && (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/>
                    </svg>
                  )}
                  <div style={{
                    padding: '6px 16px', borderRadius: 'var(--radius-md)',
                    background: `${color}12`, border: `1px solid ${color}30`,
                    fontSize: isFormula ? '0.85rem' : '0.78rem',
                    fontWeight: isFormula ? 700 : 500,
                    color: isFormula ? color : 'var(--text-secondary)',
                    fontFamily: isFormula ? "'JetBrains Mono', monospace" : 'inherit',
                  }}>
                    {label}
                  </div>
                </React.Fragment>
              ))}
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '10px', lineHeight: 1.6 }}>
              First-order differences capture movement direction, magnitude, and temporal displacement across the 32-frame sequence.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
