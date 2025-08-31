import cv2
import numpy as np
import mediapipe as mp
import time
import argparse

class MediaPipeInvisibleSkin:
    def __init__(self):
        """Initialize MediaPipe-based invisible skin effect."""
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.background_frame = None
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("MediaPipe Invisible Skin Effect initialized!")
        print("Controls:")
        print("  Press 'b' to capture background")
        print("  Press 'q' to quit")
        print("  Press 's' to save current frame")
    
    def process_frame(self, frame):
        """Process frame using MediaPipe selfie segmentation."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get segmentation mask
        results = self.selfie_segmentation.process(rgb_frame)
        mask = results.segmentation_mask
        
        if mask is not None:
            # Convert mask to proper format
            mask = np.stack((mask,) * 3, axis=-1)
            mask = (mask * 255).astype(np.uint8)
            
            # Invert mask to get background (we want to keep person, remove background)
            mask = 255 - mask
            
            # Apply mask to create invisible effect
            if self.background_frame is not None:
                # Composite with background
                mask_norm = mask.astype(np.float32) / 255.0
                result = frame.astype(np.float32) * (1 - mask_norm) +                          self.background_frame.astype(np.float32) * mask_norm
                result = result.astype(np.uint8)
            else:
                result = frame
        else:
            result = frame
        
        return result, mask
    
    def update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def run(self):
        """Main processing loop."""
        print("Starting MediaPipe video processing...")
        
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
                
                # Show the result
                cv2.imshow('MediaPipe Invisible Effect', result)
                
                # Optional: Show mask for debugging
                if mask is not None:
                    cv2.imshow('Segmentation Mask', mask)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                cv2.imshow('MediaPipe Invisible Effect', frame)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Processing stopped.")

def main():
    try:
        effect = MediaPipeInvisibleSkin()
        effect.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
