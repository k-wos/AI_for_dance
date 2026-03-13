import os
import cv2
import time
import math
import numpy as np

from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# wyciszenie logów
os.environ["YOLO_VERBOSE"] = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------------
# MODELE
# ------------------------

yolo_model = YOLO("yolov8n.pt")

model_path = "models/pose_landmarker.task"

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

# ------------------------
# VIDEO
# ------------------------

video_path = "videos/raw/dance3.mp4"
cap = cv2.VideoCapture(video_path)

# historia pozycji do stabilnego wyboru pary
history = {}

# ------------------------
# LANDMARKER
# ------------------------

with PoseLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = yolo_model.track(frame, persist=True)

        centers = []
        boxes_data = []

        for r in results:

            if r.boxes.id is None:
                continue

            ids = r.boxes.id.int().tolist()

            for box, track_id in zip(r.boxes.xyxy, ids):

                x1,y1,x2,y2 = map(int, box)

                cx = (x1+x2)/2
                cy = (y1+y2)/2

                centers.append((track_id,cx,cy))
                boxes_data.append((track_id,x1,y1,x2,y2))

        # ------------------------
        # WYBÓR NAJBLIŻSZEJ PARY
        # ------------------------

        best_pair = None
        best_dist = 999999

        for i in range(len(centers)):
            for j in range(i+1,len(centers)):

                id1,x1,y1 = centers[i]
                id2,x2,y2 = centers[j]

                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)

                if dist < best_dist:
                    best_dist = dist
                    best_pair = (id1,id2)

        if best_pair is None:
            cv2.imshow("Dance AI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        selected_ids = set(best_pair)

        # ------------------------
        # ANALIZA WYBRANEJ PARY
        # ------------------------

        for track_id,x1,y1,x2,y2 in boxes_data:

            if track_id not in selected_ids:
                continue

            person = frame[y1:y2, x1:x2]

            if person.size == 0:
                continue

            rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(time.time()*1000)

            pose_result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            # ------------------------
            # RYSOWANIE SKELETONU
            # ------------------------

            if pose_result.pose_landmarks:

                for landmark in pose_result.pose_landmarks[0]:

                    px = int(landmark.x*(x2-x1)) + x1
                    py = int(landmark.y*(y2-y1)) + y1

                    cv2.circle(frame,(px,py),4,(0,255,0),-1)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(
                frame,
                f"Dancer {track_id}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,0),
                2
            )

        # ------------------------
        # WYŚWIETLANIE
        # ------------------------

        cv2.imshow("Dance AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
