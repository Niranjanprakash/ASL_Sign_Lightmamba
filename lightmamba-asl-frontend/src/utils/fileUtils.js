import { MAX_FILE_SIZE_MB } from './constants';

export function validateVideoFile(file) {
  if (!file) return { valid: false, error: 'No file selected.' };

  const ext = file.name.split('.').pop().toLowerCase();
  if (ext !== 'mp4') {
    return { valid: false, error: 'Only .mp4 video files are supported.' };
  }

  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB > MAX_FILE_SIZE_MB) {
    return { valid: false, error: `File size exceeds ${MAX_FILE_SIZE_MB} MB limit.` };
  }

  if (file.size === 0) {
    return { valid: false, error: 'The selected file is empty.' };
  }

  return { valid: true, error: null };
}

export function formatFileSize(bytes) {
  if (bytes < 1024)        return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function createObjectURL(file) {
  return URL.createObjectURL(file);
}

export function revokeObjectURL(url) {
  if (url) URL.revokeObjectURL(url);
}
