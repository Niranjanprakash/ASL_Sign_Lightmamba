import React from 'react';
import VideoSkeletonPlayer from './VideoSkeletonPlayer';

export default function VideoPreview({ file, objectUrl }) {
  if (!file || !objectUrl) return null;
  return <VideoSkeletonPlayer file={file} objectUrl={objectUrl} />;
}
