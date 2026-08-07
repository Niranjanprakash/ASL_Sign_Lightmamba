import React, { useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader';
import ArchitectureFlow from '../components/ArchitectureFlow';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import { getModelInfo } from '../services/api';

function InfoRow({ label, value, mono }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '9px 0', borderBottom: '1px solid var(--border-card)' }}>
      <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', fontFamily: mono ? "'JetBrains Mono', monospace" : 'inherit' }}>{value ?? '—'}</span>
    </div>
  );
}

function ComponentCard({ title, color, items, description }) {
  return (
    <div className="glass-card glass-card-hover" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
        <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: color, flexShrink: 0 }} />
        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)' }}>{title}</div>
      </div>
      {description && <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '10px' }}>{description}</p>}
      {items && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
          {items.map((item) => (
            <span key={item} style={{ padding: '2px 10px', borderRadius: '12px', background: `${color}12`, border: `1px solid ${color}25`, fontSize: '0.72rem', fontWeight: 600, color }}>{item}</span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ModelInfo() {
  const [info, setInfo]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState(null);

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <PageHeader
        label="Architecture"
        title="Model Architecture"
        subtitle="LightMamba-ASL combines MobileNetV3, MediaPipe, and Hierarchical Multi-Scale Mamba for video-level ASL recognition."
      />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '24px', alignItems: 'start' }}>

        {/* ── LEFT: Architecture diagram ─────────────── */}
        <div>
          <div className="glass-card" style={{ padding: '28px' }}>
            <div className="section-label" style={{ marginBottom: '16px' }}>Complete Pipeline</div>
            <ArchitectureFlow />
          </div>

          {/* HMS-Mamba explanation */}
          <div className="glass-card" style={{ padding: '24px', marginTop: '20px' }}>
            <div className="section-label" style={{ marginBottom: '12px' }}>Proposed Architecture</div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '12px' }}>
              Hierarchical Multi-Scale Mamba (HMS-Mamba)
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '12px', marginBottom: '16px' }}>
              {[
                { scale: 'Fine', steps: 32, color: 'var(--accent-blue)', desc: 'Small finger movements, local transitions' },
                { scale: 'Intermediate', steps: 16, color: 'var(--accent-purple)', desc: 'Hand trajectories, two-hand coordination' },
                { scale: 'Global', steps: 8, color: 'var(--accent-cyan)', desc: 'Complete gesture evolution, overall structure' },
              ].map(({ scale, steps, color, desc }) => (
                <div key={scale} style={{ padding: '14px', borderRadius: 'var(--radius-md)', background: `${color}10`, border: `1px solid ${color}25`, textAlign: 'center' }}>
                  <div style={{ fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color, marginBottom: '4px' }}>{scale}</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', lineHeight: 1 }}>{steps}</div>
                  <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', marginTop: '4px' }}>temporal steps</div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: 1.4 }}>{desc}</div>
                </div>
              ))}
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
              Instead of processing the complete sign sequence at a single temporal resolution, HMS-Mamba
              learns fine-grained, intermediate, and global temporal representations simultaneously.
              The three representations are fused to produce the final video-level embedding.
            </p>
          </div>

          {/* MediaPipe explanation */}
          <div className="glass-card" style={{ padding: '24px', marginTop: '20px' }}>
            <div className="section-label" style={{ marginBottom: '12px' }}>Skeletal Representation</div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '10px' }}>
              MediaPipe Skeletal Representation
            </h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: '14px' }}>
              MediaPipe extracts structured hand and upper-body landmark coordinates from each sampled video frame.
              Instead of relying only on RGB appearance, LightMamba-ASL uses these landmarks to explicitly represent
              hand geometry, finger configuration, body-relative position, and movement across time.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '10px' }}>
              {[
                { label: 'Left Hand', count: '21', sub: 'x, y, z per point', color: 'var(--accent-blue)' },
                { label: 'Right Hand', count: '21', sub: 'x, y, z per point', color: 'var(--accent-purple)' },
                { label: 'Pose', count: '33', sub: 'x, y, z per point', color: 'var(--accent-cyan)' },
              ].map(({ label, count, sub, color }) => (
                <div key={label} style={{ padding: '12px', borderRadius: 'var(--radius-md)', background: `${color}10`, border: `1px solid ${color}25`, textAlign: 'center' }}>
                  <div style={{ fontSize: '1.4rem', fontWeight: 800, color }}>{count}</div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-primary)' }}>{label}</div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '2px' }}>{sub}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── RIGHT: Model info + components ─────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

          {/* Backend model info */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div className="section-label" style={{ marginBottom: '12px' }}>Backend Model Info</div>
            {loading && <LoadingSpinner label="Fetching model info…" />}
            {error   && <ErrorMessage message={error} />}
            {info && (
              <div>
                <InfoRow label="Model Name"       value={info.model_name} />
                <InfoRow label="Classes"          value={info.num_classes} />
                <InfoRow label="Frames / Video"   value={info.frames_per_video} />
                <InfoRow label="Input Resolution" value={info.input_resolution} />
                <InfoRow label="Motion Features"  value={info.use_motion ? 'Enabled' : 'Disabled'} />
                <InfoRow label="Checkpoint"       value={info.checkpoint_status?.checkpoint_exists ? 'Loaded' : 'Not Found'} />
              </div>
            )}
            {!loading && !info && !error && (
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Backend offline — model info unavailable.</p>
            )}
          </div>

          {/* Component cards */}
          <ComponentCard
            title="MobileNetV3-Small"
            color="var(--accent-blue)"
            description="Lightweight CNN backbone for frame-level spatial feature extraction. Pretrained on ImageNet. Outputs [T, 576] feature vectors."
            items={['Pretrained ImageNet', 'Frame-level', '[T, 576] output', 'Frozen → Fine-tuned']}
          />
          <ComponentCard
            title="MediaPipe Holistic"
            color="var(--accent-purple)"
            description="Extracts left hand, right hand, and upper-body pose landmarks from each frame. Missing landmarks are zero-filled with validity masks."
            items={['21 hand landmarks', '33 pose landmarks', 'Validity mask', 'Normalized coords']}
          />
          <ComponentCard
            title="Multimodal Fusion"
            color="var(--accent-cyan)"
            description="Combines RGB appearance features with skeletal landmark embeddings. Optional reliability-aware gating for ablation experiments."
            items={['Concat + Projection', 'Reliability gates', '[T, 256] output', 'Configurable']}
          />
          <ComponentCard
            title="HMS-Mamba"
            color="var(--accent-green)"
            description="Hierarchical Multi-Scale Mamba processes the fused sequence at three temporal resolutions. Fine (32), Intermediate (16), Global (8)."
            items={['Fine T=32', 'Intermediate T=16', 'Global T=8', 'Multi-scale fusion']}
          />
        </div>
      </div>
    </div>
  );
}
