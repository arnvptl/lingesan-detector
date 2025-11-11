import cv2
import mediapipe as mp
import numpy as np
import pygame

# Setup
pygame.mixer.init()
pygame.mixer.music.load("song.mp3")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

playing = False
hunch_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    is_hunched = False
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Get key points
        nose = (lm[0].x * w, lm[0].y * h)
        left_shoulder = (lm[11].x * w, lm[11].y * h)
        right_shoulder = (lm[12].x * w, lm[12].y * h)
        left_hip = (lm[23].x * w, lm[23].y * h)
        right_hip = (lm[24].x * w, lm[24].y * h)
        
        # Calculate midpoints
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2,
                       (left_shoulder[1] + right_shoulder[1]) / 2)
        mid_hip = ((left_hip[0] + right_hip[0]) / 2,
                  (left_hip[1] + right_hip[1]) / 2)
        
        # Calculate angle
        v1 = np.array(nose) - np.array(mid_shoulder)
        v2 = np.array(mid_hip) - np.array(mid_shoulder)
        angle = np.degrees(np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        ))
        
        # Check if hunched (angle less than 150 degrees)
        is_hunched = angle < 150
        
        # Draw skeleton
        cv2.line(frame, tuple(map(int, mid_shoulder)), tuple(map(int, mid_hip)), (0, 255, 255), 2)
        cv2.line(frame, tuple(map(int, nose)), tuple(map(int, mid_shoulder)), (255, 255, 0), 2)
        cv2.putText(frame, f"Angle: {angle:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Update hunch counter
    hunch_count = hunch_count + 1 if is_hunched else 0
    
    # Control music (need 6 consecutive hunched frames)
    if hunch_count >= 6 and not playing:
        pygame.mixer.music.play(-1)
        playing = True
    elif hunch_count == 0 and playing:
        pygame.mixer.music.pause()
        playing = False
    
    # Display status
    status = "Lingesan DETECTED!" if playing else "OK"
    color = (0, 0, 255) if playing else (0, 200, 0)
    cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    cv2.imshow("Lingesan Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
pose.close()
