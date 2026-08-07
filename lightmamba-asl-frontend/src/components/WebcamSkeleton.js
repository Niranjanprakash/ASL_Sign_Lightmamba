import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useCamera }     from '../hooks/useCamera';
import { useMediaPipe }  from '../hooks/useMediaPipe';
import { LandmarkStatus, SkeletonLegend } from './LandmarkStatus';
import ErrorMessage from './ErrorMessage';

const CameraIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
    <circle cx="12" cy="13" r="4"/>
  </svg>
);
const StopIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
  </svg>
);

export default function WebcamSkeleton({ showControls = true }) {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const wrapRef   = useRef(null);

  const [skeletonOn, setSkeletonOn] = useState(true);
  const [fps, setFps]               = useState(0);
  const fpsRef = useRef({ count: 0, last: performance.now() });

  const { active, error: camError, startCamera, stopCamera } = useCamera();
  const { tracking, mpReady, mpError } = useMediaPipe(videoRef, canvasRef, active && skeletonOn);

  /* FPS counter */
  useEffect(() => {
    if (!active) { setFps(0); return; }
    const id = setInterval(() => {
      const now     = performance.now();
      const elapsed = (now - fpsRef.current.last) / 1000;
      setFps(Math.round(fpsRef.current.count / elapsed));
      fpsRef.current = { count: 0, last: now };
    }, 1000);
    return () => clearInterval(id);
  }, [active]);

  /* Keep canvas pixel dimensions in sync with the wrapper div */
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ro = new ResizeObserver(() => {
      canvas.width  = wrap.offsetWidth;
      canvas.height = wrap.offsetHeight;
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  const handleStart = useCallback(() => startCamera(videoRef.current), [startCamera]);
  const handleStop  = useCallback(() => stopCamera(videoRef.current), [stopCamera]);

  return (
    <div>
      {/* ── Camera viewport ─────────────────────────── */}
      <div
        ref={wrapRef}
        className="webcam-container"
        style={{ position: 'relative', overflow: 'hidden' }}
      >
        {/* Video mirrored via CSS only — canvas is NOT mirrored */}
        <video
          ref={videoRef}
          playsInline
          muted
          autoPlay
          style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block', transform: 'scaleX(-1)' }}
        />

        {/* Canvas sits on top — mirrored via scaleX(-1) to match mirrored video display */}
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute', top: 0, left: 0,
            width: '100%', height: '100%',
            display: skeletonOn ? 'block' : 'none',
            pointerEvents: 'none',
            transform: 'scaleX(-1)',
          }}
        />

        {!active && (
          <div className="webcam-placeholder">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
            <p>Camera inactive — click Start Camera</p>
          </div>
        )}

        {active && (
          <>
            <div className="webcam-overlay-badge">
              <span className="pulse-dot green" />
              {skeletonOn && mpReady ? 'Skeleton Active' : 'Camera Active'}
            </div>
            <div className="webcam-fps-badge">{fps} FPS</div>
          </>
        )}
      </div>

      {/* ── Controls ────────────────────────────────── */}
      {showControls && (
        <div className="live-controls">
          {!active ? (
            <button className="btn btn-primary btn-sm" onClick={handleStart}>
              <CameraIcon /> Start Camera
            </button>
          ) : (
            <button className="btn btn-danger btn-sm" onClick={handleStop}>
              <StopIcon /> Stop Camera
            </button>
          )}

          <button
            className={`btn btn-sm ${skeletonOn ? 'btn-secondary' : 'btn-ghost'}`}
            onClick={() => setSkeletonOn((s) => !s)}
            disabled={!active}
          >
            {skeletonOn ? 'Hide Skeleton' : 'Show Skeleton'}
          </button>

          {!mpReady && !mpError && active && (
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Loading MediaPipe…
            </span>
          )}
        </div>
      )}

      {camError && <div style={{ marginTop: '10px' }}><ErrorMessage message={camError} /></div>}
      {mpError  && <div style={{ marginTop: '10px' }}><ErrorMessage message={mpError} /></div>}

      {/* ── Tracking panels ─────────────────────────── */}
      <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <LandmarkStatus tracking={tracking} />
        <SkeletonLegend />
      </div>
    </div>
  );
}
