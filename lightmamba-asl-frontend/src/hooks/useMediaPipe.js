import { useState, useRef, useCallback, useEffect } from 'react';
import { initMediaPipe, detectFrame, isReady } from '../services/mediapipeService';
import { drawSkeleton, syncCanvasToVideo, clearCanvas } from '../utils/canvasUtils';

export function useMediaPipe(videoRef, canvasRef, enabled) {
  const [mpReady, setMpReady]       = useState(false);
  const [mpError, setMpError]       = useState(null);
  const [tracking, setTracking]     = useState({
    leftHand: null, rightHand: null, pose: null,
    leftCount: 0, rightCount: 0, poseCount: 0,
  });

  const rafRef      = useRef(null);
  const runningRef  = useRef(false);

  // Initialize MediaPipe once
  useEffect(() => {
    let cancelled = false;
    initMediaPipe()
      .then(() => { if (!cancelled) setMpReady(true); })
      .catch((e) => { if (!cancelled) setMpError(`MediaPipe init failed: ${e.message}`); });
    return () => { cancelled = true; };
  }, []);

  const processLoop = useCallback(() => {
    if (!runningRef.current) return;

    const video  = videoRef.current;
    const canvas = canvasRef.current;

    if (
      video &&
      canvas &&
      isReady() &&
      !video.paused &&
      !video.ended &&
      video.readyState >= 2
    ) {
      syncCanvasToVideo(canvas, video);
      const ts     = performance.now();
      const result = detectFrame(video, ts);
      setTracking(result);
      drawSkeleton(canvas, result);
    }

    rafRef.current = requestAnimationFrame(processLoop);
  }, [videoRef, canvasRef]);

  const startTracking = useCallback(() => {
    if (runningRef.current) return;
    runningRef.current = true;
    rafRef.current = requestAnimationFrame(processLoop);
  }, [processLoop]);

  const stopTracking = useCallback(() => {
    runningRef.current = false;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (canvasRef.current) clearCanvas(canvasRef.current);
    setTracking({ leftHand: null, rightHand: null, pose: null,
                  leftCount: 0, rightCount: 0, poseCount: 0 });
  }, [canvasRef]);

  // Auto-start/stop based on enabled flag
  useEffect(() => {
    if (enabled && mpReady) {
      startTracking();
    } else {
      stopTracking();
    }
    return () => stopTracking();
  }, [enabled, mpReady, startTracking, stopTracking]);

  return { mpReady, mpError, tracking, startTracking, stopTracking };
}
