import React, { useRef, useState } from 'react';
import { validateVideoFile } from '../utils/fileUtils';
import ErrorMessage from './ErrorMessage';

const UploadIcon = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
);

const VideoIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
  </svg>
);

export default function VideoUploader({ onFileSelected }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);

  function handleFile(file) {
    setError(null);
    const { valid, error: err } = validateVideoFile(file);
    if (!valid) { setError(err); return; }
    onFileSelected(file);
  }

  function onInputChange(e) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = '';
  }

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  return (
    <div>
      <div
        className={`upload-zone${dragOver ? ' drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload MP4 video"
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,video/mp4"
          onChange={onInputChange}
          style={{ display: 'none' }}
        />
        <div className="upload-icon"><UploadIcon /></div>
        <div className="upload-title">Upload ASL Video</div>
        <div className="upload-subtitle">Drag &amp; drop your MP4 here, or click to browse</div>
        <div className="upload-hint" style={{ marginTop: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
          <VideoIcon />
          <span>MP4 format · Max 100 MB · Complete gesture video</span>
        </div>
      </div>
      {error && <div style={{ marginTop: '10px' }}><ErrorMessage message={error} onDismiss={() => setError(null)} /></div>}
    </div>
  );
}
