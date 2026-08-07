import React, { forwardRef } from 'react';

const SkeletonCanvas = forwardRef(function SkeletonCanvas({ visible = true, style }, ref) {
  return (
    <canvas
      ref={ref}
      className="skeleton-canvas-overlay"
      style={{ display: visible ? 'block' : 'none', ...style }}
      aria-hidden="true"
    />
  );
});

export default SkeletonCanvas;
