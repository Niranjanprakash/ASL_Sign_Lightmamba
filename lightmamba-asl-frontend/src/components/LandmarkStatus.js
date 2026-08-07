import React from 'react';
import { SKELETON_COLORS } from '../utils/constants';

// ── SVG icons ──────────────────────────────────────────────
const CheckIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);
const XIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);
const HandIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/><path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2"/>
    <path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>
  </svg>
);
const PersonIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="5" r="2"/><path d="M12 7v5"/><path d="M9 12H7a2 2 0 0 0-2 2v1"/><path d="M15 12h2a2 2 0 0 1 2 2v1"/>
    <path d="M9 19l-2 2"/><path d="M15 19l2 2"/>
  </svg>
);

// ── LandmarkStatus ─────────────────────────────────────────
export function LandmarkStatus({ tracking }) {
  const { leftHand, rightHand, pose, leftCount, rightCount, poseCount } = tracking || {};

  const items = [
    { label: 'Left Hand',   detected: !!leftHand,  count: leftCount,  color: SKELETON_COLORS.LEFT_HAND,  Icon: HandIcon },
    { label: 'Right Hand',  detected: !!rightHand, count: rightCount, color: SKELETON_COLORS.RIGHT_HAND, Icon: HandIcon },
    { label: 'Upper Pose',  detected: !!pose,      count: poseCount,  color: SKELETON_COLORS.POSE,       Icon: PersonIcon },
  ];

  const total = (leftCount || 0) + (rightCount || 0) + (poseCount || 0);

  return (
    <div className="tracking-card">
      <div className="tracking-card-title">Landmark Tracking</div>
      {items.map(({ label, detected, count, color, Icon }) => (
        <div key={label} className="tracking-item">
          <div className="tracking-item-left">
            <span style={{ color }}><Icon /></span>
            {label}
          </div>
          <div className={`tracking-item-right ${detected ? 'tracking-detected' : 'tracking-missing'}`}>
            {detected ? <CheckIcon /> : <XIcon />}
            {detected ? (
              <span className="landmark-count-badge">{count} pts</span>
            ) : (
              <span style={{ fontSize: '0.72rem' }}>Not Detected</span>
            )}
          </div>
        </div>
      ))}
      <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--border-card)', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
        <span style={{ color: 'var(--text-muted)' }}>Total Landmarks</span>
        <span style={{ fontWeight: 700, color: total > 0 ? 'var(--accent-blue)' : 'var(--text-muted)' }}>{total} / 75</span>
      </div>
    </div>
  );
}

// ── SkeletonLegend ─────────────────────────────────────────
export function SkeletonLegend() {
  const items = [
    { color: SKELETON_COLORS.LEFT_HAND,  label: 'Left Hand (21 landmarks)' },
    { color: SKELETON_COLORS.RIGHT_HAND, label: 'Right Hand (21 landmarks)' },
    { color: SKELETON_COLORS.POSE,       label: 'Upper Body Pose (33 landmarks)' },
  ];
  return (
    <div className="tracking-card">
      <div className="tracking-card-title">Skeleton Legend</div>
      <div className="skeleton-legend">
        {items.map(({ color, label }) => (
          <div key={label} className="legend-item">
            <div className="legend-line" style={{ background: color }} />
            <div className="legend-dot" style={{ background: color }} />
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── HandTrackingStatus ─────────────────────────────────────
export function HandTrackingStatus({ leftDetected, rightDetected, leftCount, rightCount }) {
  return (
    <div className="tracking-card">
      <div className="tracking-card-title">Hand Detection</div>
      <div className="tracking-item">
        <div className="tracking-item-left" style={{ color: SKELETON_COLORS.LEFT_HAND }}>
          <HandIcon /> Left Hand
        </div>
        <div className={`tracking-item-right ${leftDetected ? 'tracking-detected' : 'tracking-missing'}`}>
          {leftDetected ? <><CheckIcon /><span className="landmark-count-badge">{leftCount}</span></> : <><XIcon /><span style={{ fontSize: '0.72rem' }}>—</span></>}
        </div>
      </div>
      <div className="tracking-item">
        <div className="tracking-item-left" style={{ color: SKELETON_COLORS.RIGHT_HAND }}>
          <HandIcon /> Right Hand
        </div>
        <div className={`tracking-item-right ${rightDetected ? 'tracking-detected' : 'tracking-missing'}`}>
          {rightDetected ? <><CheckIcon /><span className="landmark-count-badge">{rightCount}</span></> : <><XIcon /><span style={{ fontSize: '0.72rem' }}>—</span></>}
        </div>
      </div>
    </div>
  );
}

// ── PoseTrackingStatus ─────────────────────────────────────
export function PoseTrackingStatus({ detected, count }) {
  return (
    <div className="tracking-card">
      <div className="tracking-card-title">Pose Detection</div>
      <div className="tracking-item">
        <div className="tracking-item-left" style={{ color: SKELETON_COLORS.POSE }}>
          <PersonIcon /> Upper Body
        </div>
        <div className={`tracking-item-right ${detected ? 'tracking-detected' : 'tracking-missing'}`}>
          {detected ? <><CheckIcon /><span className="landmark-count-badge">{count} pts</span></> : <><XIcon /><span style={{ fontSize: '0.72rem' }}>Not Detected</span></>}
        </div>
      </div>
    </div>
  );
}
