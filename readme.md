# Lingesan Detection AI

A fun computer vision project that detects **“Lingesan-style” hunchback posture** (inspired by *I Movie*) using your webcam — and automatically **plays or pauses music** based on your posture.

When you bend forward like Lingesan → the music starts 🎵  
When you sit straight → the music pauses ⏸️  

---

## Features
- Real-time **posture detection** using [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose)
- Auto **music play/pause** via [pygame](https://www.pygame.org/)
- Simple calibration for your own upright posture
- Works with any webcam and any song (MP3/WAV)

---

## Requirements
Install dependencies:
```bash
pip install mediapipe opencv-python numpy pygame
