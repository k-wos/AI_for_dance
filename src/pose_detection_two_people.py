import cv2
import mediapipe as mp
import csv
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- KONFIGURACJA ---
MODEL_PATH = "models/pose_landmarker_multi.task"
VIDEO_PATH = "videos/raw/dance1.mp4"
OUTPUT_CSV = "taniec_dane.csv"

# Inicjalizacja MediaPipe
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=2,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Tracker: przechowuje ostatnie dane osób {id: {"landmarks": [], "center": (x,y), "lost_frames": 0}}
trackers = {
    0: {"landmarks": None, "center": None, "lost_frames": 0, "color": (0, 255, 0)},
    1: {"landmarks": None, "center": None, "lost_frames": 0, "color": (255, 0, 0)}
}

def get_center(landmarks, w, h):
    # Środek bioder (punkt 23 i 24) jest najstabilniejszy w tańcu
    cx = int(((landmarks[23].x + landmarks[24].x) / 2) * w)
    cy = int(((landmarks[23].y + landmarks[24].y) / 2) * h)
    return (cx, cy)

# Przygotowanie pliku CSV
header = ['frame', 'person_id']
for i in range(33):
    header.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z', f'lm{i}_vis'])

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            h, w, _ = frame.shape
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            current_dets = []
            if result.pose_landmarks:
                for pose in result.pose_landmarks:
                    current_dets.append({"lms": pose, "center": get_center(pose, w, h)})

            # DOPASOWANIE (Główna logika)
            used_dets = set()
            used_tids = set()
            
            # 1. Próbuj dopasować obecne wykrycia do tego co pamiętamy
            matches = []
            for d_idx, det in enumerate(current_dets):
                for t_id, t_data in trackers.items():
                    if t_data["center"] is not None:
                        dist = ((det["center"][0]-t_data["center"][0])**2 + (det["center"][1]-t_data["center"][1])**2)**0.5
                        matches.append((dist, d_idx, t_id))
            
            matches.sort() # Najbliższe pary pierwsze

            for d, d_idx, t_id in matches:
                if d_idx not in used_dets and t_id not in used_tids and d < 200:
                    trackers[t_id]["landmarks"] = current_dets[d_idx]["lms"]
                    trackers[t_id]["center"] = current_dets[d_idx]["center"]
                    trackers[t_id]["lost_frames"] = 0
                    used_dets.add(d_idx)
                    used_tids.add(t_id)

            # 2. Jeśli ktoś został, a tracker jest wolny - przypisz
            for d_idx, det in enumerate(current_dets):
                if d_idx not in used_dets:
                    for t_id in trackers:
                        if t_id not in used_tids:
                            trackers[t_id]["landmarks"] = det["lms"]
                            trackers[t_id]["center"] = det["center"]
                            trackers[t_id]["lost_frames"] = 0
                            used_tids.add(t_id)
                            break

            # RYSOWANIE I ZAPIS
            for t_id, t_data in trackers.items():
                if t_id not in used_tids:
                    t_data["lost_frames"] += 1
                
                if t_data["landmarks"] is not None:
                    # Jeśli osoba zniknęła (lost_frames > 0), rysuj na szaro (duch)
                    is_lost = t_data["lost_frames"] > 0
                    color = (128, 128, 128) if is_lost else t_data["color"]
                    
                    # Zapis do CSV (tylko jeśli osoba jest aktualnie widziana)
                    if not is_lost:
                        row = [frame_idx, t_id]
                        for lm in t_data["landmarks"]:
                            row.extend([lm.x, lm.y, lm.z, lm.visibility])
                        writer.writerow(row)

                    # Rysowanie na ekranie
                    for lm in t_data["landmarks"]:
                        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, color, -1)
                    
                    status = "WIDOCZNY" if not is_lost else f"ZGUBIONY ({t_data['lost_frames']})"
                    cv2.putText(frame, f"ID {t_id}: {status}", (20, 40 + t_id*30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Ocena Jakosci Tanca - Tracking", frame)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()