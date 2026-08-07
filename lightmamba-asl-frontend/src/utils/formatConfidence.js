import { CONFIDENCE_LEVELS } from './constants';

export function formatPercent(value) {
  if (value == null || isNaN(value)) return '—';
  return `${(value * 100).toFixed(2)}%`;
}

export function getConfidenceLevel(confidence) {
  if (confidence >= CONFIDENCE_LEVELS.HIGH.min)   return CONFIDENCE_LEVELS.HIGH;
  if (confidence >= CONFIDENCE_LEVELS.MEDIUM.min) return CONFIDENCE_LEVELS.MEDIUM;
  return CONFIDENCE_LEVELS.LOW;
}

export function getConfidenceFillClass(confidence) {
  if (confidence >= 0.75) return 'high';
  if (confidence >= 0.50) return 'medium';
  return 'low';
}
