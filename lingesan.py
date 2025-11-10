import os, sys, json, cv2, mediapipe as mp, numpy as np, pygame

SONG_PATH = "song.mp3"
ANGLE_DROP_THRESHOLD = 18.0
NOSE_FORWARD_RATIO = 0.20
HOLD_FRAMES = 6
SMOOTH_WINDOW = 6
CALIBRATION_FILE = "lingesan_calib.json"

if not os.path.exists(SONG_PATH):
    print(f"Song not found: {SONG_PATH}"); sys.exit(1)

try: pygame.mixer.quit()
except: pass
pygame.mixer.init()
pygame.mixer.music.load(SONG_PATH)
pygame.mixer.music.stop()
pygame.mixer.music.set_volume(1.0)
player_state = "stopped"

def do_play():
    global player_state
    if player_state == "paused":
        pygame.mixer.music.unpause()
    elif player_state == "stopped" or not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)
    player_state = "playing"

def do_pause():
    global player_state
    if player_state == "playing" and pygame.mixer.music.get_busy():
        pygame.mixer.music.pause(); player_state = "paused"

def do_stop():
    global player_state
    pygame.mixer.music.stop(); player_state = "stopped"

def save_calib(c): open(CALIBRATION_FILE,"w").write(json.dumps(c))
def load_calib():
    if os.path.exists(CALIBRATION_FILE):
        try: return json.load(open(CALIBRATION_FILE))
        except: return None
    return None

def midpoint(a,b): return (np.array(a)+np.array(b))/2
def angle_at(a,b,c):
    a,b,c=np.array(a),np.array(b),np.array(c)
    ba,bc=a-b,c-b; na,nb=np.linalg.norm(ba)+1e-8,np.linalg.norm(bc)+1e-8
    cosang=np.clip(np.dot(ba,bc)/(na*nb),-1,1)
    return np.degrees(np.arccos(cosang))
def perp_dist(p,a,b):
    p,a,b=np.array(p),np.array(a),np.array(b); ab=b-a
    if np.linalg.norm(ab)<1e-8: return np.linalg.norm(p-a)
    return abs(np.cross(ab,p-a))/np.linalg.norm(ab)

mp_pose=mp.solutions.pose; pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
cap=cv2.VideoCapture(0)
if not cap.isOpened(): print("Camera failed"); sys.exit(1)

calib=load_calib() or {}; baseline=calib.get("baseline_angle")
ang_hist,rat_hist=[],[]; count=0; mode=False
print("Lingesan Detector: press 'c' to calibrate, 'q' to quit.")

try:
    while True:
        ok,frame=cap.read()
        if not ok: break
        h,w=frame.shape[:2]
        res=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        ang,rat=None,None
        if res.pose_landmarks:
            lm=res.pose_landmarks.landmark
            n=(lm[0].x*w,lm[0].y*h)
            lsh,rsh=(lm[11].x*w,lm[11].y*h),(lm[12].x*w,lm[12].y*h)
            lhp,rhp=(lm[23].x*w,lm[23].y*h),(lm[24].x*w,lm[24].y*h)
            ms, mh=midpoint(lsh,rsh), midpoint(lhp,rhp)
            ang=angle_at(n,ms,mh)
            d=perp_dist(n,ms,mh); rat=d/(np.linalg.norm(np.array(ms)-np.array(mh))+1e-8)
            cv2.line(frame,tuple(map(int,ms)),tuple(map(int,mh)),(0,255,255),2)
            cv2.line(frame,tuple(map(int,n)),tuple(map(int,ms)),(255,255,0),2)
        if ang is not None:
            ang_hist.append(ang); 
            if len(ang_hist)>SMOOTH_WINDOW: ang_hist.pop(0)
            ang_sm=np.mean(ang_hist)
        else: ang_sm=None
        if rat is not None:
            rat_hist.append(rat); 
            if len(rat_hist)>SMOOTH_WINDOW: rat_hist.pop(0)
            rat_sm=np.mean(rat_hist)
        else: rat_sm=None
        is_hunch=False
        if ang_sm is not None:
            if baseline is None and ang_sm<150: is_hunch=True
            elif baseline and (baseline-ang_sm)>=ANGLE_DROP_THRESHOLD: is_hunch=True
        if rat_sm and rat_sm>NOSE_FORWARD_RATIO: is_hunch=True
        count = count+1 if is_hunch else 0
        prev=mode; mode = count>=HOLD_FRAMES if is_hunch else False
        if mode and not prev: do_play()
        elif not mode and prev: do_pause()
        cv2.putText(frame,"LINGESAN" if mode else "OK",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255) if mode else (0,200,0),3)
        if ang_sm is not None: cv2.putText(frame,f"A:{ang_sm:.1f}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        if rat_sm is not None: cv2.putText(frame,f"N:{rat_sm:.2f}",(10,110),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.imshow("Lingesan",frame)
        k=cv2.waitKey(1)&0xFF
        if k in [ord('q'),27]: break
        elif k==ord('c') and ang_sm is not None:
            baseline=float(ang_sm); save_calib({"baseline_angle":baseline})
finally:
    do_stop(); cap.release(); cv2.destroyAllWindows()
    try: pose.close()
    except: pass
    try: pygame.mixer.quit()
    except: pass
    print("Exited cleanly.")
