"""
Gesture Detector Module
Detects hand gestures (FIST, PALM, SOS) using MediaPipe and OpenCV.
"""

import cv2
import mediapipe as mp
import time
from typing import Tuple, Optional, List


class GestureDetector:
    """
    Detects hand gestures from webcam feed using MediaPipe Hands.
    Recognizes FIST, PALM, and SOS (palm held for 3 seconds).
    """
    
    def __init__(self):
        """Initialize MediaPipe Hands model and drawing utilities."""
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Detect single hand for simplicity
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture state tracking
        self.palm_start_time: Optional[float] = None
        self.sos_trigger_duration = 3.0  # 3 seconds for SOS
        self.current_gesture = "NONE"
        
    def _is_fist(self, landmarks: List) -> bool:
        """
        Determine if hand gesture is a FIST.
        
        A fist is detected when all fingers are closed (tips below joints).
        
        Args:
            landmarks: List of hand landmark coordinates
            
        Returns:
            True if gesture is a fist, False otherwise
        """
        # Finger tip indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        # Finger pip (proximal interphalangeal) joint indices
        finger_pips = [3, 6, 10, 14, 18]
        
        fingers_closed = 0
        
        # Check thumb (special case - compare x coordinate)
        if landmarks[finger_tips[0]].x < landmarks[finger_pips[0]].x:
            fingers_closed += 1
        
        # Check other fingers (compare y coordinate)
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y > landmarks[finger_pips[i]].y:
                fingers_closed += 1
        
        # If 4 or more fingers are closed, it's a fist
        return fingers_closed >= 4
    
    def _is_palm(self, landmarks: List) -> bool:
        """
        Determine if hand gesture is a PALM (open hand).
        
        A palm is detected when all fingers are extended (tips above joints).
        
        Args:
            landmarks: List of hand landmark coordinates
            
        Returns:
            True if gesture is a palm, False otherwise
        """
        # Finger tip indices
        finger_tips = [4, 8, 12, 16, 20]
        # Finger pip joint indices
        finger_pips = [3, 6, 10, 14, 18]
        
        fingers_extended = 0
        
        # Check thumb (special case - compare x coordinate)
        if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
            fingers_extended += 1
        
        # Check other fingers (compare y coordinate)
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                fingers_extended += 1
        
        # If 4 or more fingers are extended, it's a palm
        return fingers_extended >= 4
    
    def detect_gesture(self, frame) -> Tuple[str, bool]:
        """
        Detect gesture from the current frame.
        
        Args:
            frame: Input frame from webcam (BGR format)
            
        Returns:
            Tuple of (gesture_name, is_sos_triggered)
            - gesture_name: "FIST", "PALM", or "NONE"
            - is_sos_triggered: True if SOS is detected (palm held for 3 seconds)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Reset gesture state
        self.current_gesture = "NONE"
        is_sos = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Get landmark list
                landmarks = hand_landmarks.landmark
                
                # Detect gesture type
                if self._is_palm(landmarks):
                    self.current_gesture = "PALM"
                    current_time = time.time()
                    
                    # Start timer if palm just detected
                    if self.palm_start_time is None:
                        self.palm_start_time = current_time
                    
                    # Check if palm held for SOS duration
                    elapsed_time = current_time - self.palm_start_time
                    if elapsed_time >= self.sos_trigger_duration:
                        is_sos = True
                
                elif self._is_fist(landmarks):
                    self.current_gesture = "FIST"
                    # Reset palm timer when fist detected
                    self.palm_start_time = None
                
                else:
                    # Reset palm timer for other gestures
                    self.palm_start_time = None
        
        else:
            # No hand detected, reset timer
            self.palm_start_time = None
        
        return self.current_gesture, is_sos
    
    def draw_gesture_info(self, frame, gesture: str, is_sos: bool):
        """
        Draw gesture information and SOS status on the frame.
        
        Args:
            frame: Frame to draw on
            gesture: Current gesture name
            is_sos: Whether SOS is triggered
        """
        # Display current gesture
        gesture_text = f"Gesture: {gesture}"
        cv2.putText(frame, gesture_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display palm hold time if palm is detected
        if gesture == "PALM" and self.palm_start_time is not None:
            elapsed_time = time.time() - self.palm_start_time
            time_text = f"Palm Time: {elapsed_time:.1f}s / {self.sos_trigger_duration}s"
            cv2.putText(frame, time_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display SOS alert if triggered
        if is_sos:
            # Red background for SOS
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # SOS text
            sos_text = "SOS ALERT!"
            text_size = cv2.getTextSize(sos_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, sos_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    
    def release(self):
        """Release resources."""
        self.hands.close()

