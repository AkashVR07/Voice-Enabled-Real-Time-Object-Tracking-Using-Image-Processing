# Multimodal AI: Real-Time Multi-Object Tracking with Intelligent Voice Feedback

# 🚀 Project Objective
To develop a real-time AI-powered system that combines object detection with voice interaction, making monitoring more interactive and hands-free.

# 📌 Overview
This project demonstrates a **multimodal AI system** that bridges computer vision and voice interaction. It detects and tracks multiple objects in real-time from a webcam feed, assigns persistent unique IDs to each object, and provides intelligent voice announcements when new objects enter the scene.

# Key Features
- **🎯 Real-Time Multi-Object Tracking** — YOLOv8x + ByteTrack for persistent ID assignment
- **🎙️ Intelligent Voice Feedback** — Smart announcements with cooldown mechanism (no spam)
- **⚡ Optimized Performance** — 30+ FPS on consumer hardware with CUDA acceleration
- **🎮 Interactive Controls** — Toggle audio on/off with 's' key, real-time metrics overlay
- **⚙️ Configurable** — YAML configuration for easy parameter tuning
- **Press 's' to toggle audio feedback | 'q' to quit**

# Prerequisites
- Python 3.8 or higher
- Webcam

# Installationn
1. Install dependencies - ```pip install -r requirements.txt```
2. Run the application - ```python src/main.py```

# requirements.txt
* torch>=2.0.0
* torchvision>=0.15.0
* opencv-python>=4.8.0
* ultralytics>=8.0.0
* supervision>=0.18.0
* numpy>=1.24.0
* pyttsx3>=2.90
*  pyyaml>=6.0

# Controls
Key         Action
```q```     Quit application\
```s```     Toggle voice feedback on/off

# 🏗️ Architecture

┌─────────────────────────────────────────────────────────────────┐
│                        Webcam Input                             │
└─────────────────────────────────────────────────────────────────┘\
                                │
                                ▼\
┌─────────────────────────────────────────────────────────────────┐
│                    YOLOv8 Object Detection                      │
│                   (Real-time frame processing)                  │
└─────────────────────────────────────────────────────────────────┘\
                                │
                                ▼\
┌─────────────────────────────────────────────────────────────────┐
│                     ByteTrack Tracker                           │
│            (Persistent ID assignment across frames)             │
└─────────────────────────────────────────────────────────────────┘\
                                │
                ┌───────────────┴───────────────┐\
                ▼                               ▼\
┌───────────────────────────┐     ┌───────────────────────────┐
│   Visual Annotation       │     │   Audio Feedback          │
│   • Bounding boxes        │     │   • New object detection  │
│   • Unique IDs            │     │   • Cooldown mechanism    │
│   • Confidence scores     │     │   • Natural language      │
└───────────────────────────┘     └───────────────────────────┘

# 🛠️ Tech Stack
Category              Technologies
Deep Learning         PyTorch, YOLOv8
Computer Vision	      OpenCV, Supervision
Tracking	            ByteTrack
Audio	                pyttsx3
Configuration	        YAML
Language              Python 3.8+

# 🎯 Use Cases
* Assistive Technology — Scene understanding for visually impaired users
* Surveillance — Automated monitoring with audio alerts
* Robotics — Human-robot interaction foundation
* Smart Homes — Occupancy tracking and awareness

# 🏁 Final Conclusion
This project shows how computer vision + speech technology can be combined for smarter, more accessible monitoring systems like assistive tools, surveillance, and smart environments.
