"""
Women Safety Gesture Alert System - Main Application
Real-time hand gesture detection system that triggers alerts on SOS gesture.

Usage:
    python main.py

Controls:
    - Press 'q' to quit
    - Show FIST gesture: No action
    - Show PALM gesture: Starts timer
    - Hold PALM for 3 seconds: Triggers SOS alert with alarm sound
"""

import cv2
import sys
from gesture_detector import GestureDetector
from alert_system import AlertSystem


class SafetyApp:
    """
    Main application class that coordinates gesture detection and alert system.
    """
    
    def __init__(self):
        """Initialize the safety application."""
        self.gesture_detector = GestureDetector()
        self.alert_system = AlertSystem()
        self.cap = None
        self.running = False
        
    def initialize_camera(self) -> bool:
        """
        Initialize webcam capture.
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        try:
            # Try to open default camera (index 0)
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                print("Error: Could not open webcam.")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully.")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def run(self):
        """Run the main application loop."""
        # Initialize camera
        if not self.initialize_camera():
            print("Failed to initialize camera. Exiting...")
            return
        
        self.running = True
        print("\n" + "="*50)
        print("Women Safety Gesture Alert System")
        print("="*50)
        print("\nInstructions:")
        print("- Show FIST: No action")
        print("- Show PALM: Starts timer")
        print("- Hold PALM for 3 seconds: Triggers SOS alert")
        print("- Press 'q' to quit")
        print("\nStarting detection...\n")
        
        previous_sos_state = False
        
        try:
            while self.running:
                # Read frame from webcam
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Could not read frame from camera.")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect gesture
                gesture, is_sos = self.gesture_detector.detect_gesture(frame)
                
                # Trigger alert if SOS detected
                if is_sos and not previous_sos_state:
                    self.alert_system.trigger_alert()
                elif not is_sos and previous_sos_state:
                    self.alert_system.stop_alert_sound()
                
                previous_sos_state = is_sos
                
                # Draw gesture information on frame
                self.gesture_detector.draw_gesture_info(frame, gesture, is_sos)
                
                # Add instructions text
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Women Safety Gesture Alert System', frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting application...")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
        except Exception as e:
            print(f"\nError during execution: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        
        # Stop alert system
        self.alert_system.release()
        
        # Release gesture detector
        self.gesture_detector.release()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        print("Cleanup complete. Goodbye!")


def main():
    """Main entry point."""
    app = SafetyApp()
    app.run()


if __name__ == "__main__":
    main()

