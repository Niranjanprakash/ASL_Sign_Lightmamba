import React, { useRef, useEffect, useState } from 'react';
import { useMediaPipe }  from '../hooks/useMediaPipe';
import { formatFileSize } from '../utils/fileUtils';
import { LandmarkStatus } from './LandmarkStatus';

const PlayIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="5 3 19 12 5 21 5 3"/>
  </svg>
);
const PauseIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
  </svg>
);
const EyeIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
  </svg>
);
const EyeOffIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>
    <line x1="1" y1="1" x2="23" y2="23"/>
  </svg>
);

export default function VideoSkeletonPlayer({ file, objectUrl }) {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const wrapRef   = useRef(null);

  const [playing, setPlaying]           = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [duration, setDuration]         = useState(null);

  const { tracking, mpReady, mpError } = useMediaPipe(videoRef, canvasRef, showSkeleton);

  /* Keep canvas pixel dimensions in sync with wrapper */
  useEffect(() => {
    const wrap   = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ro = new ResizeObserver(() => {
      canvas.width  = wrap.offsetWidth;
      canvas.height = wrap.offsetHeight;
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  function togglePlay() {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); }
    else          { v.pause(); setPlaying(false); }
  }

  return (
    <div>
      {/* ── Video + Canvas overlay ───────────────────── */}
      <div
        ref={wrapRef}
        style={{ position: 'relative', borderRadius: 'var(--radius-lg)', overflow: 'hidden', background: '#000', aspectRatio: '16/9' }}
      >
        <video
          ref={videoRef}
          src={objectUrl}
          onLoadedMetadata={(e) => setDuration(e.target.duration)}
          onEnded={() => setPlaying(false)}
          playsInline
          style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute', top: 0, left: 0,
            width: '100%', height: '100%',
            display: showSkeleton ? 'block' : 'none',
            pointerEvents: 'none',
          }}
        />
      </div>

      {/* ── Controls ────────────────────────────────── */}
      <div className="video-controls">
        <button className="btn btn-secondary btn-sm" onClick={togglePlay}>
          {playing ? <PauseIcon /> : <PlayIcon />}
          {playing ? 'Pause' : 'Play'}
        </button>

        <button
          className={`btn btn-sm ${showSkeleton ? 'btn-primary' : 'btn-ghost'}`}
          onClick={() => setShowSkeleton((s) => !s)}
          disabled={!!mpError}
          title={mpError || (mpReady ? 'Toggle skeleton overlay' : 'MediaPipe loading…')}
        >
          {showSkeleton ? <EyeOffIcon /> : <EyeIcon />}
          {showSkeleton ? 'Hide Skeleton' : 'Show Skeleton'}
        </button>

        {!mpReady && !mpError && (
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Loading MediaPipe…</span>
        )}
        {mpError && (
          <span style={{ fontSize: '0.72rem', color: 'var(--accent-amber)' }}>Skeleton unavailable</span>
        )}
      </div>

      {/* ── File info bar ───────────────────────────── */}
      <div className="video-info-bar" style={{ marginTop: '10px' }}>
        <div className="video-info-item">
          <span>File:</span>
          <strong>{file?.name}</strong>
        </div>
        <div className="video-info-item">
          <span>Size:</span>
          <strong>{formatFileSize(file?.size || 0)}</strong>
        </div>
        {duration != null && (
          <div className="video-info-item">
            <span>Duration:</span>
            <strong>{duration.toFixed(2)}s</strong>
          </div>
        )}
        {showSkeleton && (
          <div className="video-info-item">
            <span className="pulse-dot green" style={{ marginRight: '4px' }} />
            <strong style={{ color: 'var(--accent-green)' }}>MediaPipe Active</strong>
          </div>
        )}
      </div>

      {/* ── Landmark status when skeleton is on ─────── */}
      {showSkeleton && (
        <div style={{ marginTop: '12px' }}>
          <LandmarkStatus tracking={tracking} />
        </div>
      )}
    </div>
  );
}
