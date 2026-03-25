import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
import torch
import pyttsx3
from collections import Counter, defaultdict
import threading
import time
from datetime import datetime

# ==================== CONFIGURATION ====================
MODEL = "yolov8x.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model
model = YOLO(MODEL).to(device)
model.fuse()  # Optimize for inference

# Class names dictionary
CLASS_NAMES_DICT = model.model.names

# Selected classes to track (Fixed duplicate 76)
selected_classes = [0, 2, 3, 5, 15, 16, 24, 25, 28, 39, 44, 56, 63, 64, 66, 67, 73, 76]
# Class names for reference:
# 0: person, 2: car, 3: motorcycle, 5: bus, 15: Cat, 16: Dog, 24: backpack, 25: umbrella, 73: book,
# 28 : tie, 39: bottle, 44: wine glass, 56: chair, 63: laptop, 64: mouse, 66: keyboard, 67: Cell phone, 76: scissors

# Initialize annotators
box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.WHITE)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=5)

# Initialize ByteTrack tracker for multi-object tracking
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30
)

# ==================== AUDIO CONFIGURATION ====================
engine = pyttsx3.init()
engine.setProperty('rate', 150)     # Speech rate (words per minute)
engine.setProperty('volume', 0.9)   # Volume level (0.0 to 1.0)

# Get available voices (optional - uncomment to select a different voice)
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)  # Use female voice if available

class AudioFeedbackManager:
    """Manages audio feedback with intelligent throttling to avoid spam"""
    
    def __init__(self, engine, cooldown_seconds=5):
        self.engine = engine
        self.cooldown_seconds = cooldown_seconds
        self.last_spoken_time = defaultdict(float)
        self.new_objects = Counter()
        self.lock = threading.Lock()
        
    def register_new_object(self, class_id, tracker_id):
        """Register a newly detected object"""
        with self.lock:
            key = f"{class_id}_{tracker_id}"
            current_time = time.time()
            
            # Only speak if cooldown period has passed for this specific object
            if current_time - self.last_spoken_time[key] >= self.cooldown_seconds:
                self.new_objects[class_id] += 1
                self.last_spoken_time[key] = current_time
                return True
        return False
    
    def speak_summary(self):
        """Speak summary of newly detected objects"""
        with self.lock:
            if self.new_objects:
                # Create natural language summary
                object_names = []
                for class_id, count in self.new_objects.items():
                    name = CLASS_NAMES_DICT[class_id]
                    if count == 1:
                        object_names.append(f"a {name}")
                    else:
                        object_names.append(f"{count} {name}s")
                
                summary = f"Detected {', '.join(object_names)}"
                
                # Speak in a separate thread to not block
                def speak():
                    self.engine.say(summary)
                    self.engine.runAndWait()
                
                threading.Thread(target=speak, daemon=True).start()
                
                # Clear the counter
                self.new_objects.clear()
                
                # Print to console for debugging
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {summary}")

# ==================== VIDEO PROCESSING ====================
def process_frame(frame, model, tracker, audio_manager, prev_counts, frame_count):
    """Process a single frame with detection, tracking, and annotation"""
    
    # Resize for faster processing (maintain aspect ratio)
    height, width = frame.shape[:2]
    scale_factor = 0.5  # Reduce to 50% for inference
    small_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
    
    try:
        # Run YOLO inference
        results = model(small_frame, verbose=False)[0]
        
        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Scale coordinates back to original frame size
        if len(detections.xyxy) > 0:
            detections.xyxy /= scale_factor
        
        # Filter by selected classes
        if len(detections.class_id) > 0:
            mask = np.isin(detections.class_id, selected_classes)
            detections = detections[mask]
            
            # Update tracker to get persistent IDs
            if len(detections.xyxy) > 0:
                detections = tracker.update_with_detections(detections)
                
                # Count current objects by class and track ID
                current_counts = Counter()
                tracked_objects = {}
                
                for idx, (class_id, tracker_id) in enumerate(zip(detections.class_id, detections.tracker_id)):
                    if tracker_id is not None:
                        current_counts[class_id] += 1
                        tracked_objects[tracker_id] = class_id
                        
                        # Check for new objects (not seen before)
                        if tracker_id not in prev_counts:
                            audio_manager.register_new_object(class_id, tracker_id)
                
                # Update previous counts with current frame's objects
                new_prev_counts = {}
                for tracker_id, class_id in tracked_objects.items():
                    new_prev_counts[tracker_id] = class_id
                
                # Create labels with tracker ID and confidence
                labels = []
                for idx, (class_id, confidence, tracker_id) in enumerate(zip(
                    detections.class_id, detections.confidence, detections.tracker_id
                )):
                    class_name = CLASS_NAMES_DICT[class_id]
                    if tracker_id is not None:
                        labels.append(f"#{tracker_id}: {class_name} ({confidence:.2f})")
                    else:
                        labels.append(f"{class_name} ({confidence:.2f})")
                
                # Annotate frame
                annotated_frame = box_annotator.annotate(frame, detections)
                annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
                
                # Add FPS and object count overlay
                cv2.putText(
                    annotated_frame, 
                    f"Active Tracks: {len(tracked_objects)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                return annotated_frame, tracked_objects
            
        # If no detections, return original frame
        return frame, {}
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, {}

# ==================== AUDIO THREAD FUNCTION ====================
def audio_thread_func(audio_manager, stop_event, interval=4):
    """Background thread to handle periodic audio summaries"""
    while not stop_event.is_set():
        audio_manager.speak_summary()
        time.sleep(interval)

# ==================== MAIN FUNCTION ====================
def main():
    """Main function to run the voice-enabled object tracking system"""
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam!")
        return
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Webcam initialized: {frame_width}x{frame_height} at {fps} FPS")
    print(f"Tracking selected classes: {[CLASS_NAMES_DICT[c] for c in selected_classes]}")
    print("Press 'q' to quit, 's' to toggle audio feedback")
    
    # Initialize audio manager
    audio_manager = AudioFeedbackManager(engine, cooldown_seconds=3)
    
    # Setup audio thread
    stop_event = threading.Event()
    audio_thread = threading.Thread(target=audio_thread_func, args=(audio_manager, stop_event))
    audio_thread.daemon = True
    audio_thread.start()
    
    # Tracking variables
    prev_tracked_objects = {}
    frame_count = 0
    audio_enabled = True
    start_time = time.time()
    
    # Create window
    cv2.namedWindow('Voice-Enabled Real-Time Object Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Voice-Enabled Real-Time Object Tracking', 1280, 720)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to capture frame")
                break
            
            frame_count += 1
            
            # Process the frame
            annotated_frame, current_tracked_objects = process_frame(
                frame, model, tracker, audio_manager, prev_tracked_objects, frame_count
            )
            
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                current_fps = frame_count / elapsed_time
                cv2.putText(
                    annotated_frame,
                    f"FPS: {current_fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Display audio status
            audio_status = "Audio: ON" if audio_enabled else "Audio: OFF"
            cv2.putText(
                annotated_frame,
                audio_status,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if audio_enabled else (0, 0, 255),
                2
            )
            
            # Display instructions
            cv2.putText(
                annotated_frame,
                "Press 'q' to quit | 's' to toggle audio",
                (10, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Show the frame
            cv2.imshow('Voice-Enabled Real-Time Object Tracking', annotated_frame)
            
            # Update previous tracked objects
            prev_tracked_objects = current_tracked_objects.copy()
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                audio_enabled = not audio_enabled
                if audio_enabled:
                    print("Audio feedback enabled")
                    # Speak confirmation
                    engine.say("Audio feedback enabled")
                    engine.runAndWait()
                else:
                    print("Audio feedback disabled")
                    engine.say("Audio feedback disabled")
                    engine.runAndWait()
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        print("Cleaning up...")
        stop_event.set()
        if audio_thread.is_alive():
            audio_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()
        engine.stop()
        print("Application terminated")

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    print("=" * 60)
    print("VOICE-ENABLED REAL-TIME OBJECT TRACKING SYSTEM")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    main()