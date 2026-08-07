/**
 * mediapipeService.js
 *
 * Loads @mediapipe/tasks-vision from CDN via a <script> tag so webpack
 * never tries to bundle the WASM module. This eliminates:
 *   - "the request of a dependency is an expression" warning
 *   - missing vision_bundle_mjs.js.map source-map error
 */

const WASM_PATH =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';

const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

let handLandmarker = null;
let poseLandmarker  = null;
let initPromise     = null;
let mediaPipeTasksVision = null;

/** Dynamically imports the ESM vision bundle from CDN. */
async function loadCDNScript() {
  if (mediaPipeTasksVision) return mediaPipeTasksVision;
  try {
    mediaPipeTasksVision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');
    return mediaPipeTasksVision;
  } catch (e) {
    throw new Error('Failed to load MediaPipe CDN script: ' + e.message);
  }
}

/**
 * Initialises HandLandmarker and PoseLandmarker once.
 * Safe to call multiple times — returns the same promise.
 */
export async function initMediaPipe() {
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const { FilesetResolver, HandLandmarker, PoseLandmarker } = await loadCDNScript();

    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence:  0.5,
      minTrackingConfidence:      0.5,
    });

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: POSE_MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence:  0.5,
      minTrackingConfidence:      0.5,
    });
  })();

  return initPromise;
}

/**
 * Runs hand + pose detection on a single video frame.
 * @param {HTMLVideoElement} frame
 * @param {number} timestampMs
 */
export function detectFrame(frame, timestampMs) {
  if (!handLandmarker || !poseLandmarker) {
    return { leftHand: null, rightHand: null, pose: null,
             leftCount: 0, rightCount: 0, poseCount: 0 };
  }

  let leftHand  = null;
  let rightHand = null;
  let pose      = null;

  try {
    const handResult = handLandmarker.detectForVideo(frame, timestampMs);
    if (handResult.landmarks && handResult.handedness) {
      handResult.landmarks.forEach((lms, idx) => {
        const side = handResult.handedness[idx]?.[0]?.categoryName;
        // MediaPipe returns mirrored labels for front-facing camera
        if (side === 'Left')  rightHand = lms;
        if (side === 'Right') leftHand  = lms;
      });
    }
  } catch (_) { /* frame not ready */ }

  try {
    const poseResult = poseLandmarker.detectForVideo(frame, timestampMs);
    if (poseResult.landmarks?.length > 0) {
      pose = poseResult.landmarks[0];
    }
  } catch (_) { /* frame not ready */ }

  return {
    leftHand,
    rightHand,
    pose,
    leftCount:  leftHand  ? leftHand.length  : 0,
    rightCount: rightHand ? rightHand.length : 0,
    poseCount:  pose      ? pose.length      : 0,
  };
}

export function isReady() {
  return handLandmarker !== null && poseLandmarker !== null;
}

export async function closeMediaPipe() {
  if (handLandmarker) { handLandmarker.close(); handLandmarker = null; }
  if (poseLandmarker) { poseLandmarker.close();  poseLandmarker  = null; }
  initPromise = null;
}
