import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import csv
from collections import defaultdict

from src import Annotator, ViewTransformer, SpeedEstimator
from config import (
    IN_VIDEO_PATH,
    OUT_VIDEO_PATH,
    YOLO_MODEL_PATH,
    LINE_Y,
    SOURCE_POINTS,
    TARGET_POINTS,
    WINDOW_NAME,
)

# =====================================================
# 1. INITIALISATION
# =====================================================

# Load YOLO (VisDrone)
model = YOLO(YOLO_MODEL_PATH)
print("Model classes:", model.names)

# Video info
video_info = sv.VideoInfo.from_video_path(IN_VIDEO_PATH)
print(video_info)

# Tracker ByteTrack
tracker = sv.ByteTrack(frame_rate=video_info.fps)

# Line zone
offset = 50
start = sv.Point(offset, LINE_Y)
end = sv.Point(video_info.width - offset, LINE_Y)

line_zone = sv.LineZone(
    start,
    end,
    minimum_crossing_threshold=1,
    triggering_anchors=(sv.Position.CENTER,)
)

# Perspective transform
SOURCE = np.array(SOURCE_POINTS, dtype=np.float32)
TARGET = np.array(TARGET_POINTS, dtype=np.float32)
view_transformer = ViewTransformer(SOURCE, TARGET)

# Speed estimator
speed_estimator = SpeedEstimator(
    fps=video_info.fps,
    view_transformer=view_transformer
)

# Annotator
annotator = Annotator(
    resolution_wh=video_info.resolution_wh,
    box_annotator=True,
    label_annotator=True,
    line_annotator=True,
    trace_annotator=True,
    polygon_zone=SOURCE,
)

# Window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, video_info.width, video_info.height)



# =====================================================
# 2. CLASSES VISDRONE
# =====================================================

VEHICLE_CLASS_IDS = [3, 4, 5, 8, 9]
track_class_map = {}

# =====================================================
# 3. STRUCTURES POUR FEATURES
# =====================================================

# Historique des données pour chaque véhicule
vehicle_history = defaultdict(lambda: {
    'speeds': [],
    'positions': [],
    'frames': []
})

# CSV pour sauvegarder les features
csv_file = open('vehicle_features.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'frame',
    'tracker_id',
    'vehicle_class',
    'speed',
    'acceleration',
    'distance_to_nearest',
    'direction_change',
    'avg_traffic_speed',
    'relative_speed'
])

# =====================================================
# 4. FONCTIONS D'EXTRACTION DE FEATURES
# =====================================================

def calculate_acceleration(tracker_id, current_speed):
    """Calcule l'accélération (différence de vitesse)"""
    history = vehicle_history[tracker_id]['speeds']
    
    if len(history) < 2:
        return 0.0
    
    # Accélération = différence vitesse actuelle - vitesse précédente
    prev_speed = history[-1]
    acceleration = current_speed - prev_speed
    
    return round(acceleration, 2)

def calculate_distance_to_nearest(detections, current_idx):
    """Calcule la distance au véhicule le plus proche"""
    if len(detections) <= 1:
        return 999.0  # Pas d'autre véhicule
    
    # Position du véhicule actuel (centre de la bounding box)
    current_box = detections.xyxy[current_idx]
    current_center = np.array([
        (current_box[0] + current_box[2]) / 2,
        (current_box[1] + current_box[3]) / 2
    ])
    
    min_distance = 999.0
    
    # Calculer distance avec tous les autres véhicules
    for i, box in enumerate(detections.xyxy):
        if i == current_idx:
            continue
        
        other_center = np.array([
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2
        ])
        
        distance = np.linalg.norm(current_center - other_center)
        min_distance = min(min_distance, distance)
    
    return round(min_distance, 2)

def calculate_direction_change(tracker_id, current_position):
    """Calcule le changement de direction (variation d'angle)"""
    history = vehicle_history[tracker_id]['positions']
    
    if len(history) < 2:
        return 0.0
    
    # Prendre les 2 dernières positions
    prev_pos = np.array(history[-1])
    prev_prev_pos = np.array(history[-2]) if len(history) >= 2 else prev_pos
    current_pos = np.array(current_position)
    
    # Vecteurs de direction
    vec1 = prev_pos - prev_prev_pos
    vec2 = current_pos - prev_pos
    
    # Éviter division par zéro
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 < 0.1 or norm2 < 0.1:
        return 0.0
    
    # Calculer l'angle entre les deux vecteurs
    cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    return round(angle, 2)

def calculate_avg_traffic_speed(detections):
    """Calcule la vitesse moyenne du trafic"""
    if 'speed' not in detections.data or len(detections.data['speed']) == 0:
        return 0.0
    
    speeds = detections.data['speed']
    valid_speeds = speeds[speeds > 0]
    
    if len(valid_speeds) == 0:
        return 0.0
    
    return round(np.mean(valid_speeds), 2)

# =====================================================
# 5. BOUCLE PRINCIPALE
# =====================================================

frame_generator = sv.get_video_frames_generator(IN_VIDEO_PATH)
frame_count = 0

with sv.VideoSink(OUT_VIDEO_PATH, video_info) as sink:
    for frame in frame_generator:
        frame_count += 1

        # -------- YOLO DETECTION --------
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filtrage VisDrone
        mask = np.isin(detections.class_id, VEHICLE_CLASS_IDS)
        detections = detections[mask]

        if len(detections) == 0:
            sink.write_frame(frame)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # -------- TRACKING --------
        detections = tracker.update_with_detections(detections)

        # -------- CLASSE STABLE --------
        class_names = []
        for tid, cid in zip(detections.tracker_id, detections.class_id):
            if tid not in track_class_map:
                track_class_map[tid] = model.names[cid]
            class_names.append(track_class_map[tid])

        detections.data["class_name"] = np.array(class_names)

        # -------- COUNTING --------
        line_zone.trigger(detections)

        # -------- SPEED --------
        detections = speed_estimator.update(detections)

        # -------- CALCUL VITESSE MOYENNE TRAFIC --------
        avg_traffic_speed = calculate_avg_traffic_speed(detections)

        # -------- EXTRACTION DES FEATURES --------
        for idx, (tid, cname, speed) in enumerate(zip(
            detections.tracker_id,
            detections.data["class_name"],
            detections.data["speed"]
        )):
            # Position du véhicule (centre de bbox)
            box = detections.xyxy[idx]
            current_position = [
                (box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2
            ]
            
            # 1. ACCÉLÉRATION
            acceleration = calculate_acceleration(tid, speed)
            
            # 2. DISTANCE AU PLUS PROCHE
            distance_to_nearest = calculate_distance_to_nearest(detections, idx)
            
            # 3. CHANGEMENT DE DIRECTION
            direction_change = calculate_direction_change(tid, current_position)
            
            # 4. VITESSE RELATIVE (différence avec moyenne trafic)
            relative_speed = round(speed - avg_traffic_speed, 2) if speed > 0 else 0.0
            
            # Sauvegarder historique
            vehicle_history[tid]['speeds'].append(speed)
            vehicle_history[tid]['positions'].append(current_position)
            vehicle_history[tid]['frames'].append(frame_count)
            
            # Limiter historique à 10 dernières frames
            if len(vehicle_history[tid]['speeds']) > 10:
                vehicle_history[tid]['speeds'].pop(0)
                vehicle_history[tid]['positions'].pop(0)
                vehicle_history[tid]['frames'].pop(0)
            
            # ÉCRIRE DANS CSV
            csv_writer.writerow([
                frame_count,
                tid,
                cname,
                speed,
                acceleration,
                distance_to_nearest,
                direction_change,
                avg_traffic_speed,
                relative_speed
            ])

        # -------- LABELS --------
        labels = []
        for tid, cname, speed in zip(
            detections.tracker_id,
            detections.data["class_name"],
            detections.data["speed"]
        ):
            if speed > 0:
                labels.append(f"{cname} #{tid} | {speed} km/h")
            else:
                labels.append(f"{cname} #{tid}")

        # -------- ANNOTATION --------
        annotated_frame = annotator.annotate(
            frame=frame,
            detections=detections,
            labels=labels,
            line_zones=[line_zone],
        )

        # -------- OUTPUT --------
        sink.write_frame(annotated_frame)
        cv2.imshow(WINDOW_NAME, annotated_frame)

        if (
            cv2.waitKey(1) & 0xFF == ord("q")
            or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        ):
            break

# Fermer le fichier CSV
csv_file.close()

cv2.destroyAllWindows()

# =====================================================
# 6. RESULTATS
# =====================================================

print("Processing complete.")
print(f"Output video: {OUT_VIDEO_PATH}")
print(f"Features saved to: vehicle_features.csv")
print(f"Total vehicles: {line_zone.in_count + line_zone.out_count}")
print(f"In: {line_zone.in_count} | Out: {line_zone.out_count}")