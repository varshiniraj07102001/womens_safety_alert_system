"""
Alert System Module
Handles alarm sounds and visual alerts when SOS is detected.
"""

import pygame
import threading
import time
import math
import numpy as np
from typing import Optional


class AlertSystem:
    """
    Manages alarm sounds and alert notifications.
    Plays alarm sound when SOS is triggered.
    """
    
    def __init__(self):
        """Initialize pygame mixer for sound playback."""
        # Initialize pygame mixer
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Generate a simple alarm tone programmatically
        self._generate_alarm_sound()
        
        # Alert state
        self.is_alerting = False
        self.alert_thread: Optional[threading.Thread] = None
        self.stop_alert = False
    
    def _generate_alarm_sound(self):
        """
        Generate a simple alarm sound using pygame.
        Creates a beeping sound that alternates between two frequencies.
        """
        try:
            # Create a simple beep sound
            sample_rate = 22050
            duration = 0.3  # 300ms per beep
            frequency1 = 800  # First tone frequency
            frequency2 = 1000  # Second tone frequency
            
            # Generate first tone
            frames1 = int(duration * sample_rate)
            arr1 = np.zeros((frames1, 2), dtype=np.int16)
            for i in range(frames1):
                wave = int(4096 * math.sin(2.0 * math.pi * frequency1 * i / sample_rate))
                arr1[i] = [wave, wave]
            
            # Generate second tone
            frames2 = int(duration * sample_rate)
            arr2 = np.zeros((frames2, 2), dtype=np.int16)
            for i in range(frames2):
                wave = int(4096 * math.sin(2.0 * math.pi * frequency2 * i / sample_rate))
                arr2[i] = [wave, wave]
            
            # Combine tones to create alternating beep pattern
            sound_array = np.concatenate([arr1, arr2, arr1, arr2])
            self.alarm_sound = pygame.sndarray.make_sound(sound_array)
            
        except Exception as e:
            print(f"Warning: Could not generate alarm sound: {e}")
            print("Alert system will work but without sound.")
            self.alarm_sound = None
    
    def _play_alarm_loop(self):
        """
        Play alarm sound in a loop until stopped.
        Runs in a separate thread to avoid blocking.
        """
        while not self.stop_alert and self.is_alerting:
            if self.alarm_sound:
                try:
                    self.alarm_sound.play()
                    # Wait for sound to finish before playing again
                    time.sleep(0.9)  # Slightly longer than sound duration
                except Exception as e:
                    print(f"Error playing alarm: {e}")
                    break
            else:
                # If no sound, just wait
                time.sleep(0.5)
    
    def trigger_alert(self):
        """
        Trigger the SOS alert (alarm sound).
        Can be called multiple times safely.
        """
        if not self.is_alerting:
            self.is_alerting = True
            self.stop_alert = False
            
            # Start alarm in separate thread
            if self.alert_thread is None or not self.alert_thread.is_alive():
                self.alert_thread = threading.Thread(target=self._play_alarm_loop, daemon=True)
                self.alert_thread.start()
                print("ALERT: SOS detected! Alarm activated.")
    
    def stop_alert_sound(self):
        """
        Stop the alarm sound.
        """
        if self.is_alerting:
            self.is_alerting = False
            self.stop_alert = True
            
            # Stop any playing sound
            if self.alarm_sound:
                try:
                    pygame.mixer.stop()
                except:
                    pass
            
            print("Alert stopped.")
    
    def release(self):
        """Release resources and stop any active alerts."""
        self.stop_alert_sound()
        try:
            pygame.mixer.quit()
        except:
            pass

