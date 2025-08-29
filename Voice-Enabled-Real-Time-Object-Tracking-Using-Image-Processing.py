import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
import torch
import pyttsx3
from collections import Counter
import threading
import time

# Load YOLOv8 model with GPU support if available
MODEL = "yolov8x.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL).to(device)
model.fuse()

# Dict mapping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# List of class IDs to track in the video stream
selected_classes = [0, 2, 3, 5, 23, 25, 39, 44, 56, 63, 64, 66, 76, 76]

# Create instance of BoxAnnotator for drawing bounding boxes
box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.WHITE)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)     # Set speech rate
engine.setProperty('volume', 0.9)   # Set volume level

def speak_summary(counts):
    """Generate and speak out the summary of detected objects."""
    summary = ", ".join(f"{CLASS_NAMES_DICT[class_id]} detected" for class_id in counts)
    engine.say(summary)
    engine.runAndWait()

def audio_thread(new_detections, stop_event):
    """Thread function to periodically speak out detected objects."""
    while not stop_event.is_set():
        if new_detections:
            speak_summary(new_detections)
            new_detections.clear()  
        time.sleep(1)

# Open the default laptop webcam for video capture (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam!")
    exit()

# Adjust frame size for better performance
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Start the audio thread
new_detections = Counter()
stop_event = threading.Event()
thread = threading.Thread(target=audio_thread, args=(new_detections, stop_event))
thread.daemon = True
thread.start()

# To keep track of previous counts for comparison
prev_counts = Counter()

# OpenCV window setup
cv2.namedWindow('YOLOv8 Real-time Detection', cv2.WINDOW_NORMAL)

# Main loop for processing video frames
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: No frame captured from the webcam!")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

    # Model prediction on the resized frame
    try:
        results = model(small_frame, verbose=True)[0]  # Set verbose=True for debugging
        detections = sv.Detections.from_ultralytics(results)

        # If no detections, continue to next frame
        if len(detections.xyxy) == 0:
            print("No detections found in this frame.")
            continue

        # Adjust detections to the original frame size
        detections.xyxy *= 2

        # Filter detections to only include selected classes
        detections = detections[np.isin(detections.class_id, selected_classes)]

        # Count objects by class for the current frame
        current_counts = Counter(detections.class_id)

        # Detect new objects or increases in counts
        for class_id, count in current_counts.items():
            if count > prev_counts.get(class_id, 0):
                new_detections[class_id] += count - prev_counts[class_id]

        # Update previous counts for the next iteration
        prev_counts = current_counts.copy()

        # Format custom labels with counts and confidence
        labels = [
            f"{CLASS_NAMES_DICT[class_id]} {current_counts[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate frame with bounding boxes and labels
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Ensure annotated frame is not empty
        if annotated_frame is None or annotated_frame.size == 0:
            print("Error: Annotated frame is empty!")
            continue

        # Display the annotated frame
        cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)

    except Exception as e:
        print(f"Error during detection processing: {e}")

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Signal the audio thread to stop and wait for it to finish
stop_event.set()
