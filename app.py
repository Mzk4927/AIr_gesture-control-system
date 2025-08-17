import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

class GestureController:
    def __init__(self):
        print("🚀 Initializing MediaPipe Gesture Controller...")
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Check webcam
        if not self.cap.isOpened():
            raise Exception("❌ Cannot access webcam. Please check your camera.")
            
        # Get screen dimensions for cursor mapping
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"📺 Screen resolution: {self.screen_width}x{self.screen_height}")
        
        # Gesture detection variables
        self.prev_time = 0
        self.click_cooldown = 0.5  # Seconds between clicks
        self.last_click_time = 0
        
        # Smoothing variables for cursor movement
        self.smoothing_factor = 0.3
        self.prev_x, self.prev_y = 0, 0
        
        # Gesture thresholds
        self.pinch_threshold = 40  # Distance threshold for pinch detection
        self.palm_threshold = 0.15  # Threshold for open palm detection
        
        # PyAutoGUI settings
        pyautogui.FAILSAFE = True  # Move mouse to corner to stop
        pyautogui.PAUSE = 0.01
        
        print("✅ MediaPipe Gesture Controller initialized successfully!")
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_pinch_gesture(self, landmarks):
        """Detect pinch gesture (thumb tip touching index finger tip)"""
        thumb_tip = landmarks[4]  # Thumb tip
        index_tip = landmarks[8]  # Index finger tip
        
        # Calculate distance between thumb and index finger tips
        distance = self.calculate_distance(thumb_tip, index_tip) * 1000
        
        return distance < self.pinch_threshold
    
    def is_open_palm(self, landmarks):
        """Detect open palm gesture (all fingers extended)"""
        # Check if all fingertips are above their respective PIP joints
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_pips = [6, 10, 14, 18]  # Index, Middle, Ring, Pinky PIP joints
        
        extended_fingers = 0
        
        # Check each finger (except thumb)
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:  # Tip is above PIP
                extended_fingers += 1
        
        # Check thumb separately (different orientation)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        if thumb_tip.x > thumb_ip.x:  # Thumb extended (assuming right hand)
            extended_fingers += 1
        
        return extended_fingers >= 4  # At least 4 fingers extended
    
    def get_cursor_position(self, landmarks, frame_width, frame_height):
        """Convert hand position to screen cursor position"""
        # Use index finger tip for cursor control
        index_tip = landmarks[8]
        
        # Map hand coordinates to screen coordinates
        x = int(index_tip.x * self.screen_width)
        y = int(index_tip.y * self.screen_height)
        
        # Apply smoothing
        smooth_x = int(self.prev_x + (x - self.prev_x) * self.smoothing_factor)
        smooth_y = int(self.prev_y + (y - self.prev_y) * self.smoothing_factor)
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        
        return smooth_x, smooth_y
    
    def draw_landmarks_and_info(self, frame, landmarks, gesture_text):
        """Draw hand landmarks and gesture information on frame"""
        # Draw hand landmarks
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Display gesture information
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display instructions
        instructions = [
            "🎮 MediaPipe Gesture Control",
            "👌 Pinch (thumb + index) = Left Click",
            "✋ Open Palm = Right Click", 
            "👆 Index finger = Cursor control",
            "Press 'q' to quit | Move mouse to corner for emergency stop"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, frame.shape[0] - 120 + i * 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main loop for gesture control"""
        print("\n🎮 AIR TOUCH GESTURE CONTROL STARTED!")
        print("=" * 50)
        print("Controls:")
        print("👌 Pinch (thumb + index finger) = Left Click")
        print("✋ Open Palm (5 fingers) = Right Click")  
        print("👆 Index finger position = Cursor control")
        print("🛑 Press 'q' to quit")
        print("⚠️  Move mouse to top-left corner for emergency stop")
        print("=" * 50)
        
        frame_count = 0
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("❌ Failed to read from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            gesture_text = "No hand detected"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    
                    # Get cursor position
                    cursor_x, cursor_y = self.get_cursor_position(
                        landmarks, frame_width, frame_height)
                    
                    # Move cursor
                    try:
                        pyautogui.moveTo(cursor_x, cursor_y)
                    except pyautogui.FailSafeException:
                        print("🛑 Emergency stop activated!")
                        break
                    
                    current_time = time.time()
                    
                    # Check for gestures
                    if self.is_pinch_gesture(landmarks):
                        gesture_text = "👌 Pinch - Left Click"
                        if current_time - self.last_click_time > self.click_cooldown:
                            pyautogui.click(button='left')
                            self.last_click_time = current_time
                            print("🖱️ Left click performed")
                    
                    elif self.is_open_palm(landmarks):
                        gesture_text = "✋ Open Palm - Right Click"
                        if current_time - self.last_click_time > self.click_cooldown:
                            pyautogui.click(button='right')
                            self.last_click_time = current_time
                            print("🖱️ Right click performed")
                    
                    else:
                        gesture_text = "👆 Hand detected - Cursor control"
                    
                    # Draw landmarks and info
                    self.draw_landmarks_and_info(frame, hand_landmarks, gesture_text)
            
            else:
                # No hand detected
                self.draw_landmarks_and_info(frame, None, gesture_text)
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if self.prev_time != 0 else 0
            self.prev_time = current_time
            
            cv2.putText(frame, f"FPS: {int(fps)}", (frame_width - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('🎮 MediaPipe Air Touch Gesture Control', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Exiting gesture control...")
                break
                
            frame_count += 1
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Air Touch Gesture Control stopped successfully.")

def main():
    """Main function to run the gesture controller"""
    try:
        print("🚀 Starting MediaPipe Air Touch Gesture Control...")
        controller = GestureController()
        controller.run()
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("💡 Make sure your webcam is connected and not being used by another application")
    finally:
        cv2.destroyAllWindows()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()