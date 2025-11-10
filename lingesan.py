"""
lingesan_detector_fixed.py
Robust Lingesan detector with reliable play/pause behavior.

Controls:
  c - calibrate upright baseline
  q or ESC - quit

Requirements:
  pip install mediapipe opencv-python numpy pygame
"""
import os, sys, time, json
import cv2, mediapipe as mp, numpy as np
import pygame

# ---------- CONFIG ----------
SONG_PATH = "song.mp3"
ANGLE_DROP_THRESHOLD = 18.0
NOSE_FORWARD_RATIO = 0.20
HOLD_FRAMES = 6
SMOOTH_WINDOW = 6
CALIBRATION_FILE = "lingesan_calib.json"
# ----------------------------

# --- Audio initialization (clean) ---
if not os.path.exists(SONG_PATH):
    print(f"ERROR: song file not found at {SONG_PATH}")
    sys.exit(1)

# Ensure clean mixer state
try:
    pygame.mixer.quit()
except Exception:
    pass
pygame.mixer.init()
# Pre-load, then stop to ensure consistent behavior between runs
try:
    pygame.mixer.music.load(SONG_PATH)
    pygame.mixer.music.stop()
    pygame.mixer.music.set_volume(1.0)
except Exception as e:
    print("Audio init error:", e)
    sys.exit(1)

# Player state: "stopped", "playing", "paused"
player_state = "stopped"

def do_play():
    global player_state
    try:
        if player_state == "paused":
            pygame.mixer.music.unpause()
            player_state = "playing"
            print("[AUDIO] unpaused -> playing")
        elif player_state == "playing":
            # already playing; no-op
            # but guard: if not busy, start fresh
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
                player_state = "playing"
                print("[AUDIO] was playing but not busy -> play() called")
        else:  # stopped
            # load in case not loaded
            pygame.mixer.music.play(-1)
            player_state = "playing"
            print("[AUDIO] stopped -> play() called")
    except Exception as e:
        print("Error in do_play():", e)

def do_pause():
    global player_state
    try:
        # pause only if playing
        if player_state == "playing" and pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            player_state = "paused"
            print("[AUDIO] pause() called -> paused")
        else:
            # if it's stopped or already paused, do nothing
            print(f"[AUDIO] pause() skipped (state={player_state}, busy={pygame.mixer.music.get_busy()})")
    except Exception as e:
        print("Error in do_pause():", e)

def do_stop():
    global player_state
    try:
        pygame.mixer.music.stop()
        player_state = "stopped"
        print("[AUDIO] stop() called -> stopped")
    except Exception as e:
        print("Error in do_stop():", e)

# --- Calibration load/save ---
def save_calibration(calib):
    try:
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(calib, f)
    except Exception as e:
        print("Save calib failed:", e)

def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# --- Math helpers ---
def midpoint(a, b):
    return (np.array(a) + np.array(b)) / 2.0

def angle_at_point(a, b, c):
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    na, nb = np.linalg.norm(ba)+1e-8, np.linalg.norm(bc)+1e-8
    cosang = np.clip(np.dot(ba, bc) / (na*nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def perpendicular_distance(p, a, b):
    p, a, b = np.array(p), np.array(a), np.array(b)
    ab = b - a
    if np.linalg.norm(ab) < 1e-8:
        return float(np.linalg.norm(p - a))
    return float(abs(np.cross(ab, p - a)) / np.linalg.norm(ab))

# --- MediaPipe setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera open failed")
    sys.exit(1)

# State variables
calib = load_calibration() or {}
baseline_angle = calib.get("baseline_angle", None)
angle_hist, ratio_hist = [], []
hungry_count = 0
lingesan_mode = False

print("Starting fixed Lingesan detector. Press 'c' to calibrate, 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)

        angle_deg = None
        ratio = None
        detected = False

        if res.pose_landmarks:
            detected = True
            lm = res.pose_landmarks.landmark
            nose = (lm[0].x*w, lm[0].y*h)
            l_sh = (lm[11].x*w, lm[11].y*h); r_sh = (lm[12].x*w, lm[12].y*h)
            l_hp = (lm[23].x*w, lm[23].y*h); r_hp = (lm[24].x*w, lm[24].y*h)
            mid_sh = midpoint(l_sh, r_sh)
            mid_hp = midpoint(l_hp, r_hp)

            # Angle (nose - mid_sh - mid_hp)
            angle_deg = angle_at_point(nose, mid_sh, mid_hp)
            dist = perpendicular_distance(nose, mid_sh, mid_hp)
            base = np.linalg.norm(np.array(mid_sh) - np.array(mid_hp)) + 1e-8
            ratio = dist / base

            # draw guiding lines
            cv2.line(frame, tuple(map(int, mid_sh)), tuple(map(int, mid_hp)), (0,255,255), 2)
            cv2.line(frame, tuple(map(int, nose)), tuple(map(int, mid_sh)), (255,255,0), 2)
            cv2.circle(frame, tuple(map(int, nose)), 4, (0,0,255), -1)

        # smoothing
        if angle_deg is not None:
            angle_hist.append(angle_deg)
            if len(angle_hist) > SMOOTH_WINDOW: angle_hist.pop(0)
            angle_sm = float(np.mean(angle_hist))
        else:
            angle_sm = None

        if ratio is not None:
            ratio_hist.append(ratio)
            if len(ratio_hist) > SMOOTH_WINDOW: ratio_hist.pop(0)
            ratio_sm = float(np.mean(ratio_hist))
        else:
            ratio_sm = None

        # decide hunch
        is_hunch = False
        reasons = []
        if angle_sm is not None:
            if baseline_angle is None:
                if angle_sm < 150.0:
                    is_hunch = True
                    reasons.append("angle<150")
            else:
                if (baseline_angle - angle_sm) >= ANGLE_DROP_THRESHOLD:
                    is_hunch = True
                    reasons.append(f"angle drop >= {ANGLE_DROP_THRESHOLD}")
        if ratio_sm is not None and ratio_sm > NOSE_FORWARD_RATIO:
            is_hunch = True
            reasons.append(f"nose ratio {ratio_sm:.2f}>{NOSE_FORWARD_RATIO:.2f}")

        # debounce / hold frames
        if is_hunch:
            hungry_count += 1
        else:
            hungry_count = 0

        prev_mode = lingesan_mode
        if hungry_count >= HOLD_FRAMES:
            lingesan_mode = True
        elif not is_hunch:
            lingesan_mode = False

        # Handle transitions for audio reliably
        if lingesan_mode and not prev_mode:
            # just entered lingesan mode => play or resume
            print("[STATE] Entered LINGESAN mode")
            do_play()
        elif not lingesan_mode and prev_mode:
            # just returned to OK => pause
            print("[STATE] Returned to OK posture")
            do_pause()

        # UI overlays
        color = (0,0,255) if lingesan_mode else (0,200,0)
        text = "LINGESAN DETECTED" if lingesan_mode else "OK"
        cv2.putText(frame, text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        if angle_sm is not None:
            cv2.putText(frame, f"Angle: {angle_sm:.1f}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if ratio_sm is not None:
            cv2.putText(frame, f"Nose ratio: {ratio_sm:.2f}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if baseline_angle is not None:
            cv2.putText(frame, f"Baseline: {baseline_angle:.1f}", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        if reasons:
            cv2.putText(frame, "Reasons: " + ", ".join(reasons), (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

        cv2.imshow("Lingesan Detector (fixed)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            if angle_sm is not None:
                baseline_angle = float(angle_sm)
                save_calibration({"baseline_angle": baseline_angle})
                print(f"[CALIB] Saved baseline angle = {baseline_angle:.2f}")

finally:
    do_stop()
    cap.release()
    cv2.destroyAllWindows()
    try:
        pose.close()
    except Exception:
        pass
    try:
        pygame.mixer.quit()
    except Exception:
        pass
    print("Exited cleanly.")
