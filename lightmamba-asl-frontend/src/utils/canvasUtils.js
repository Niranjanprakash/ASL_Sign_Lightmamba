import { HAND_CONNECTIONS, POSE_UPPER_CONNECTIONS, SKELETON_COLORS } from './constants';

/**
 * Clears the canvas completely.
 */
export function clearCanvas(canvas) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Draws a single set of landmarks and their connections.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} landmarks  - array of {x, y} normalized [0..1]
 * @param {Array} connections - array of [i, j] index pairs
 * @param {string} color
 * @param {number} dotRadius
 * @param {number} lineWidth
 */
function drawLandmarkSet(ctx, landmarks, connections, color, dotRadius = 4, lineWidth = 2) {
  if (!landmarks || landmarks.length === 0) return;

  const w = ctx.canvas.width;
  const h = ctx.canvas.height;

  // Draw connections
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = 0.75;
  for (const [i, j] of connections) {
    const a = landmarks[i];
    const b = landmarks[j];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(a.x * w, a.y * h);
    ctx.lineTo(b.x * w, b.y * h);
    ctx.stroke();
  }

  // Draw dots
  ctx.globalAlpha = 1.0;
  for (const lm of landmarks) {
    if (!lm) continue;
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, dotRadius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    // White center
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, dotRadius * 0.4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.fill();
  }
}

/**
 * Draws the full skeleton from a GestureRecognizer / HandLandmarker result.
 * @param {HTMLCanvasElement} canvas
 * @param {object} result - { leftHand, rightHand, pose }
 *   leftHand / rightHand: array of {x,y,z} normalized landmarks or null
 *   pose: array of {x,y,z} normalized landmarks or null
 */
export function drawSkeleton(canvas, result) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  clearCanvas(canvas);

  if (!result) return;

  const { leftHand, rightHand, pose } = result;

  if (leftHand) {
    drawLandmarkSet(ctx, leftHand, HAND_CONNECTIONS, SKELETON_COLORS.LEFT_HAND, 4, 2);
  }

  if (rightHand) {
    drawLandmarkSet(ctx, rightHand, HAND_CONNECTIONS, SKELETON_COLORS.RIGHT_HAND, 4, 2);
  }

  if (pose) {
    // Only upper-body joints
    const upperIndices = new Set(POSE_UPPER_CONNECTIONS.flat());
    const upperPose = pose.map((lm, i) => upperIndices.has(i) ? lm : null);
    drawLandmarkSet(ctx, upperPose, POSE_UPPER_CONNECTIONS, SKELETON_COLORS.POSE, 5, 2.5);
  }

  ctx.globalAlpha = 1.0;
}

/**
 * Syncs canvas intrinsic pixel dimensions to the video's rendered size.
 * Uses offsetWidth/offsetHeight so it works even when transform:scaleX(-1) is applied.
 */
export function syncCanvasToVideo(canvas, videoEl) {
  if (!canvas || !videoEl) return;
  const w = videoEl.offsetWidth;
  const h = videoEl.offsetHeight;
  if (w > 0 && h > 0 && (canvas.width !== w || canvas.height !== h)) {
    canvas.width  = w;
    canvas.height = h;
  }
}
