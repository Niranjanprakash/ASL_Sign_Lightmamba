import { useState, useRef, useCallback } from 'react';

export function useCamera() {
  const [active, setActive] = useState(false);
  const [error, setError]   = useState(null);
  const streamRef           = useRef(null);

  const startCamera = useCallback(async (videoEl) => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoEl) {
        videoEl.srcObject = stream;
        await videoEl.play();
      }
      setActive(true);
    } catch (err) {
      const msg =
        err.name === 'NotAllowedError'
          ? 'Camera permission denied. Please allow camera access in your browser.'
          : err.name === 'NotFoundError'
          ? 'No camera device found.'
          : `Camera error: ${err.message}`;
      setError(msg);
      setActive(false);
    }
  }, []);

  const stopCamera = useCallback((videoEl) => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoEl) {
      videoEl.srcObject = null;
    }
    setActive(false);
  }, []);

  return { active, error, streamRef, startCamera, stopCamera };
}
