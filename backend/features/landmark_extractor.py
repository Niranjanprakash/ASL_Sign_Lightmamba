import cv2
import numpy as np
import mediapipe as mp

class LandmarkExtractor:
    """
    Extracts Left Hand, Right Hand, and Pose landmarks using MediaPipe Holistic.
    Returns:
        - raw_landmarks: list or array of shape [T, 75, 3]
        - mask: list or array of shape [T, 3] representing visibility/presence of (left_hand, right_hand, pose)
    """
    def __init__(self):
        try:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.fallback = False
        except AttributeError:
            print("[WARNING] MediaPipe legacy 'solutions' module is not available (common on Python 3.13). Using zero-filled landmark fallback.")
            self.fallback = True

    def extract_from_frame(self, frame_rgb: np.ndarray):
        if self.fallback:
            # Return empty coordinates [75, 3] and mask [0, 0, 0]
            return np.zeros((75, 3)), np.zeros(3)
            
        results = self.holistic.process(frame_rgb)
        
        # Initialize empty coords
        lh_coords = np.zeros((21, 3))
        rh_coords = np.zeros((21, 3))
        pose_coords = np.zeros((33, 3))
        
        # Validity mask: [left_hand_detected, right_hand_detected, pose_detected]
        mask = np.zeros(3)
        
        if results.left_hand_landmarks:
            mask[0] = 1.0
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                lh_coords[i] = [lm.x, lm.y, lm.z]
                
        if results.right_hand_landmarks:
            mask[1] = 1.0
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                rh_coords[i] = [lm.x, lm.y, lm.z]
                
        if results.pose_landmarks:
            mask[2] = 1.0
            for i, lm in enumerate(results.pose_landmarks.landmark):
                pose_coords[i] = [lm.x, lm.y, lm.z]

        # Combine all landmarks: [75, 3]
        combined = np.concatenate([lh_coords, rh_coords, pose_coords], axis=0)
        return combined, mask

    def extract_video_sequence(self, frames: np.ndarray):
        """
        Frames input shape: [T, H, W, C]
        Returns:
            - sequence_landmarks: [T, 75, 3]
            - sequence_masks: [T, 3]
        """
        seq_landmarks = []
        seq_masks = []
        
        for frame in frames:
            landmarks, mask = self.extract_from_frame(frame)
            seq_landmarks.append(landmarks)
            seq_masks.append(mask)
            
        return np.stack(seq_landmarks, axis=0), np.stack(seq_masks, axis=0)

    def close(self):
        if not self.fallback:
            self.holistic.close()
