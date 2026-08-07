import React from 'react';
import WebcamSkeleton from './WebcamSkeleton';

// MediaPipeVisualizer wraps WebcamSkeleton for use in the Skeleton Demo page
export default function MediaPipeVisualizer() {
  return <WebcamSkeleton showControls={true} />;
}
