import cv2
import numpy as np
import time
import argparse
from collections import deque

class InvisibleSkinEffect:
    def __init__(self, buffer_size=3):
        """
        Initialize the invisible skin effect system.
        
        Args:
            buffer_size: Number of frames for temporal smoothing
        """
        self.buffer_size = buffer_size
        
        # Initialize variables
        self.background_frame = None
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Temporal smoothing buffer
        self.mask_buffer = deque(maxlen=buffer_size)
        
        # Simple and effective skin detection parameters
        # HSV ranges for skin detection
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # YCrCb ranges for skin detection
        self.lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        # Set webcam properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Invisible Skin Effect initialized!")
        print("Controls:")
        print("  Press 'b' to capture background")
        print("  Press 'q' to quit")
        print("  Press 's' to save current frame")
        print("  Press '1'/'2' to adjust skin detection sensitivity")
        print("  Press '3'/'4' to adjust saturation range")
        print("  Press '5'/'6' to adjust value range")
        print("  Press 'r' to reset detection parameters")
    
    def detect_skin_simple(self, frame):
        """Simple and effective skin detection."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask using HSV
        skin_mask_hsv = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Create skin mask using YCrCb
        skin_mask_ycrcb = cv2.inRange(ycrcb, self.lower_ycrcb, self.upper_ycrcb)
        
        # Combine both masks
        skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)
        
        return skin_mask
    
    def clean_mask(self, mask):
        """Clean and improve the skin mask."""
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Find contours and filter by area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a new mask with filtered contours
        clean_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                cv2.fillPoly(clean_mask, [contour], 255)
        
        return clean_mask
    
    def smooth_mask(self, mask):
        """Apply smoothing to the mask."""
        # Apply Gaussian blur for smooth edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def apply_temporal_smoothing(self, mask):
        """Apply temporal smoothing to reduce flickering."""
        self.mask_buffer.append(mask)
        
        if len(self.mask_buffer) < 2:
            return mask
        
        # Average the masks in the buffer
        smoothed_mask = np.mean(self.mask_buffer, axis=0).astype(np.uint8)
        return smoothed_mask
    
    def composite_frame(self, frame, mask, background):
        """Composite the frame with background replacement."""
        if background is None:
            return frame
        
        # Normalize mask to 0-1 range
        mask_norm = mask.astype(np.float32) / 255.0
        
        # Create 3-channel mask
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        # Composite: background * mask + frame * (1 - mask)
        # This replaces skin areas with background
        result = background.astype(np.float32) * mask_3ch + \
                 frame.astype(np.float32) * (1 - mask_3ch)
        
        return result.astype(np.uint8)
    
    def update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def reset_parameters(self):
        """Reset detection parameters to default values."""
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        self.lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        print("Parameters reset to default values")
    
    def process_frame(self, frame):
        """Process a single frame through the pipeline."""
        # Detect skin using simple method
        skin_mask = self.detect_skin_simple(frame)
        
        # Clean the mask
        clean_mask = self.clean_mask(skin_mask)
        
        # Apply smoothing
        smoothed_mask = self.smooth_mask(clean_mask)
        final_mask = self.apply_temporal_smoothing(smoothed_mask)
        
        # Composite with background
        if self.background_frame is not None:
            result = self.composite_frame(frame, final_mask, self.background_frame)
        else:
            result = frame
        
        return result, final_mask
    
    def run(self):
        """Main processing loop."""
        print("Starting video processing...")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from webcam")
                break
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('b'):
                self.background_frame = frame.copy()
                print("Background captured!")
            elif key == ord('s'):
                cv2.imwrite(f"frame_{int(time.time())}.jpg", frame)
                print("Frame saved!")
            elif key == ord('r'):
                self.reset_parameters()
            elif key == ord('1'):  # Decrease hue range
                self.lower_skin[0] = max(0, self.lower_skin[0] - 2)
                self.upper_skin[0] = max(0, self.upper_skin[0] - 2)
                print(f"Hue range decreased: {self.lower_skin[0]}-{self.upper_skin[0]}")
            elif key == ord('2'):  # Increase hue range
                self.lower_skin[0] = min(180, self.lower_skin[0] + 2)
                self.upper_skin[0] = min(180, self.upper_skin[0] + 2)
                print(f"Hue range increased: {self.lower_skin[0]}-{self.upper_skin[0]}")
            elif key == ord('3'):  # Decrease saturation range
                self.lower_skin[1] = max(0, self.lower_skin[1] - 5)
                self.upper_skin[1] = max(0, self.upper_skin[1] - 5)
                print(f"Saturation range decreased: {self.lower_skin[1]}-{self.upper_skin[1]}")
            elif key == ord('4'):  # Increase saturation range
                self.lower_skin[1] = min(255, self.lower_skin[1] + 5)
                self.upper_skin[1] = min(255, self.upper_skin[1] + 5)
                print(f"Saturation range increased: {self.lower_skin[1]}-{self.upper_skin[1]}")
            elif key == ord('5'):  # Decrease value range
                self.lower_skin[2] = max(0, self.lower_skin[2] - 5)
                self.upper_skin[2] = max(0, self.upper_skin[2] - 5)
                print(f"Value range decreased: {self.lower_skin[2]}-{self.upper_skin[2]}")
            elif key == ord('6'):  # Increase value range
                self.lower_skin[2] = min(255, self.lower_skin[2] + 5)
                self.upper_skin[2] = min(255, self.upper_skin[2] + 5)
                print(f"Value range increased: {self.lower_skin[2]}-{self.upper_skin[2]}")
            
            # Process frame
            try:
                result, mask = self.process_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Add FPS text to frame
                cv2.putText(result, f"FPS: {self.fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add status text
                status = "Background: Captured" if self.background_frame is not None else "Background: Not captured"
                cv2.putText(result, status, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add controls text
                cv2.putText(result, "1/2:Hue 3/4:Sat 5/6:Value R:Reset", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show the result
                cv2.imshow('Invisible Skin Effect', result)
                
                # Show mask for debugging
                if mask is not None:
                    cv2.imshow('Skin Mask', mask)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                cv2.imshow('Invisible Skin Effect', frame)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Processing stopped.")

def main():
    parser = argparse.ArgumentParser(description='Invisible Skin Effect')
    parser.add_argument('--buffer-size', type=int, default=3,
                       help='Number of frames for temporal smoothing')
    
    args = parser.parse_args()
    
    try:
        effect = InvisibleSkinEffect(buffer_size=args.buffer_size)
        effect.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 