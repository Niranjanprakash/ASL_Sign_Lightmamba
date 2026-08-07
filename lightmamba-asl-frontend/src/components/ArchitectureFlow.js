import React from 'react';

function FlowNode({ title, sub, color = 'var(--accent-blue)' }) {
  return (
    <div style={{
      padding: '10px 14px', borderRadius: 'var(--radius-md)',
      background: `${color}12`, border: `1px solid ${color}30`,
      textAlign: 'center', minWidth: '120px',
    }}>
      <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-primary)' }}>{title}</div>
      {sub && <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '2px' }}>{sub}</div>}
    </div>
  );
}

function Arrow({ label }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '2px 0', color: 'var(--text-muted)' }}>
      <svg width="16" height="18" viewBox="0 0 16 18" fill="none">
        <line x1="8" y1="0" x2="8" y2="12" stroke="currentColor" strokeWidth="1.5"/>
        <polyline points="4,8 8,14 12,8" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
      {label && <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>{label}</span>}
    </div>
  );
}

export default function ArchitectureFlow() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0 }}>
      <FlowNode title="ASL MP4 Video" sub="Complete gesture sequence" color="var(--accent-cyan)" />
      <Arrow />
      <FlowNode title="32-Frame Temporal Sampling" sub="Uniform temporal coverage" color="var(--accent-blue)" />
      <Arrow />

      {/* Two parallel branches */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', width: '100%', maxWidth: '500px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0 }}>
          <FlowNode title="RGB Frames" sub="224×224 px" color="var(--accent-blue)" />
          <Arrow />
          <FlowNode title="MobileNetV3-Small" sub="Pretrained ImageNet" color="var(--accent-blue)" />
          <Arrow />
          <FlowNode title="Visual Features" sub="[T, 576]" color="var(--accent-blue)" />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0 }}>
          <FlowNode title="Video Frames" sub="Same 32 frames" color="var(--accent-purple)" />
          <Arrow />
          <FlowNode title="MediaPipe" sub="Holistic landmarks" color="var(--accent-purple)" />
          <Arrow />
          <FlowNode title="75 Landmarks" sub="LH + RH + Pose" color="var(--accent-purple)" />
          <Arrow />
          <FlowNode title="Norm + Motion" sub="Δt = Lt − Lt−1" color="var(--accent-purple)" />
        </div>
      </div>

      <Arrow label="Multimodal Fusion" />
      <FlowNode title="Multimodal Fusion" sub="RGB + Skeletal → [T, 256]" color="var(--accent-cyan)" />
      <Arrow />
      <FlowNode title="Hierarchical Multi-Scale Mamba" sub="HMS-Mamba" color="var(--accent-green)" />
      <Arrow />

      {/* Three scales */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '10px', width: '100%', maxWidth: '400px' }}>
        <FlowNode title="Fine" sub="T = 32" color="var(--accent-blue)" />
        <FlowNode title="Intermediate" sub="T = 16" color="var(--accent-purple)" />
        <FlowNode title="Global" sub="T = 8" color="var(--accent-cyan)" />
      </div>

      <Arrow label="Multi-Scale Fusion" />
      <FlowNode title="Video Representation" sub="[B, 256]" color="var(--accent-green)" />
      <Arrow />
      <FlowNode title="10-Class ASL Prediction" sub="Softmax + Confidence" color="var(--accent-amber)" />
    </div>
  );
}
