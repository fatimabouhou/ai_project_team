import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import csv
from collections import defaultdict, deque

from src import ViewTransformer, SpeedEstimator
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
# PALETTE DE COULEURS MODERNE
# =====================================================
COLORS = {
    'primary': (255, 107, 107),      # Rouge corail
    'secondary': (78, 205, 196),     # Turquoise
    'accent': (255, 195, 0),         # Jaune doré
    'success': (88, 214, 141),       # Vert menthe
    'warning': (255, 159, 64),       # Orange
    'dark': (45, 52, 54),            # Gris foncé
    'light': (223, 230, 233),        # Gris clair
    'white': (255, 255, 255),
}

# =====================================================
# ANNOTATEUR PERSONNALISÉ
# =====================================================
class ModernAnnotator:
    def __init__(self, resolution_wh, polygon_zone, line_start, line_end):
        self.resolution_wh = resolution_wh
        self.polygon_zone = polygon_zone
        self.line_start = line_start
        self.line_end = line_end
        
        # Annotateurs Supervision
        self.box_annotator = sv.BoxAnnotator(
            thickness=3,
            color=sv.ColorPalette.from_hex(['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
        )
    
    def draw_overlay_panel(self, frame, line_zone, avg_speed, vehicle_count):
        """Dessine un panneau d'informations moderne et structuré en overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Panneau supérieur avec meilleure structure
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (w, panel_height), COLORS['dark'], -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        # Titre du projet - Centré et stylisé
        title = "VEHICLE TRACKING & SPEED ANALYSIS"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, COLORS['white'], 3)
        
        # Ligne de séparation décorative
        cv2.line(frame, (30, 60), (w-30, 60), COLORS['secondary'], 3)
        
        # Section statistiques principale - Organisée en grille
        stats_y = 85
        
        # Stat 1: Total Vehicles
        stat1_text = f"VEHICLES: {vehicle_count}"
        stat1_size = cv2.getTextSize(stat1_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        stat1_x = 50
        stat1_w = stat1_size[0] + 30
        cv2.rectangle(frame, (stat1_x-10, stats_y-15), (stat1_x+stat1_w, stats_y+10), COLORS['success'], -1)
        cv2.putText(frame, stat1_text, (stat1_x, stats_y+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS['white'], 2)
        
        # Stat 2: Entrées
        stat2_text = f"IN: {line_zone.in_count}"
        stat2_size = cv2.getTextSize(stat2_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        stat2_x = stat1_x + stat1_w + 40
        stat2_w = stat2_size[0] + 30
        cv2.rectangle(frame, (stat2_x-10, stats_y-15), (stat2_x+stat2_w, stats_y+10), COLORS['primary'], -1)
        cv2.putText(frame, stat2_text, (stat2_x, stats_y+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS['white'], 2)
        
        # Stat 3: Sorties
        stat3_text = f"OUT: {line_zone.out_count}"
        stat3_size = cv2.getTextSize(stat3_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        stat3_x = stat2_x + stat2_w + 40
        stat3_w = stat3_size[0] + 30
        cv2.rectangle(frame, (stat3_x-10, stats_y-15), (stat3_x+stat3_w, stats_y+10), COLORS['warning'], -1)
        cv2.putText(frame, stat3_text, (stat3_x, stats_y+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS['white'], 2)
        
        # Stat 4: Vitesse moyenne - Aligné à droite
        stat4_text = f"AVG SPEED: {avg_speed:.1f} km/h"
        stat4_size = cv2.getTextSize(stat4_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        stat4_x = w - stat4_size[0] - 50
        stat4_w = stat4_size[0] + 30
        cv2.rectangle(frame, (stat4_x-10, stats_y-15), (stat4_x+stat4_w, stats_y+10), COLORS['accent'], -1)
        cv2.putText(frame, stat4_text, (stat4_x, stats_y+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS['dark'], 2)
        
        return frame
    
    def draw_zone_polygon(self, frame):
        """Dessine la zone de transformation avec meilleure visibilité"""
        overlay = frame.copy()
        
        # Zone colorée semi-transparente - Opacité ajustée
        pts = self.polygon_zone.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], COLORS['secondary'])
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        
        # Bordure de la zone - Plus épaisse et visible
        cv2.polylines(frame, [pts], True, COLORS['secondary'], 4, lineType=cv2.LINE_AA)
        # Contour interne pour effet de profondeur
        cv2.polylines(frame, [pts], True, COLORS['white'], 2, lineType=cv2.LINE_AA)
        
        # Points de coins avec effet glow amélioré
        for point in self.polygon_zone:
            pt = tuple(point.astype(int))
            # Cercle externe (glow)
            cv2.circle(frame, pt, 10, COLORS['accent'], -1)
            # Cercle interne
            cv2.circle(frame, pt, 6, COLORS['white'], -1)
            # Bordure
            cv2.circle(frame, pt, 10, COLORS['dark'], 2, lineType=cv2.LINE_AA)
        
        return frame
    
    def draw_line_zone(self, frame, line_zone):
        """Dessine la ligne de comptage avec style moderne et clair"""
        start = (int(self.line_start.x), int(self.line_start.y))
        end = (int(self.line_end.x), int(self.line_end.y))
        
        # Ligne principale épaisse avec double contour pour visibilité
        cv2.line(frame, start, end, COLORS['primary'], 6)
        cv2.line(frame, start, end, COLORS['white'], 3)
        
        # Indicateurs de direction améliorés
        mid_x = (start[0] + end[0]) // 2
        mid_y = start[1]
        
        # Panneau IN (haut) - Plus visible
        in_arrow_start = (mid_x - 80, mid_y - 40)
        in_arrow_end = (mid_x - 80, mid_y - 15)
        cv2.arrowedLine(frame, in_arrow_start, in_arrow_end,
                       COLORS['success'], 4, tipLength=0.5, line_type=cv2.LINE_AA)
        # Fond pour texte IN
        in_text = "IN"
        in_text_size = cv2.getTextSize(in_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        in_bg_x = mid_x - 110
        in_bg_y = mid_y - 60
        cv2.rectangle(frame, (in_bg_x - 5, in_bg_y - in_text_size[1] - 5),
                     (in_bg_x + in_text_size[0] + 5, in_bg_y + 5),
                     COLORS['success'], -1)
        cv2.putText(frame, in_text, (in_bg_x, in_bg_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, COLORS['white'], 2, cv2.LINE_AA)
        
        # Panneau OUT (bas) - Plus visible
        out_arrow_start = (mid_x + 80, mid_y + 15)
        out_arrow_end = (mid_x + 80, mid_y + 40)
        cv2.arrowedLine(frame, out_arrow_start, out_arrow_end,
                       COLORS['warning'], 4, tipLength=0.5, line_type=cv2.LINE_AA)
        # Fond pour texte OUT
        out_text = "OUT"
        out_text_size = cv2.getTextSize(out_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        out_bg_x = mid_x + 50
        out_bg_y = mid_y + 60
        cv2.rectangle(frame, (out_bg_x - 5, out_bg_y - out_text_size[1] - 5),
                     (out_bg_x + out_text_size[0] + 5, out_bg_y + 5),
                     COLORS['warning'], -1)
        cv2.putText(frame, out_text, (out_bg_x, out_bg_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, COLORS['white'], 2, cv2.LINE_AA)
        
        return frame
    
    def draw_vehicle_indicators(self, frame, detections):
        """Dessine des barres de vitesse améliorées pour chaque véhicule
        Les barres des véhicules trop éloignés ne sont pas affichées"""
        for idx, box in enumerate(detections.xyxy):
            # Calculer la taille de la bounding box pour filtrer les véhicules éloignés
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            box_area = box_width * box_height
            
            # Seuil minimum: même que pour les labels
            min_box_area = 1500  # Surface minimale en pixels
            min_box_height = 40  # Hauteur minimale en pixels
            
            # Filtrer les véhicules trop éloignés
            if box_area < min_box_area or box_height < min_box_height:
                continue  # Ne pas afficher la barre pour ce véhicule
            
            # Indicateur de vitesse avec barre
            if 'speed' in detections.data and detections.data['speed'][idx] > 0:
                speed = detections.data['speed'][idx]
                
                # Barre de vitesse - position et taille améliorées
                bar_width = 80
                bar_height = 12
                bar_x = int(box[0])
                bar_y = int(box[1]) - 25
                
                # Ombre de la barre pour effet 3D
                shadow_offset = 3
                cv2.rectangle(frame, 
                            (bar_x + shadow_offset, bar_y + shadow_offset), 
                            (bar_x + bar_width + shadow_offset, bar_y + bar_height + shadow_offset),
                            (0, 0, 0), -1)
                
                # Fond de la barre (gris foncé)
                cv2.rectangle(frame, (bar_x, bar_y), 
                            (bar_x + bar_width, bar_y + bar_height),
                            COLORS['dark'], -1)
                
                # Bordure claire
                cv2.rectangle(frame, (bar_x, bar_y), 
                            (bar_x + bar_width, bar_y + bar_height),
                            COLORS['light'], 2)
                
                # Remplissage proportionnel (max 100 km/h pour meilleure échelle)
                max_speed = 100
                fill_ratio = min(speed, max_speed) / max_speed
                fill_width = int(fill_ratio * (bar_width - 4))
                
                # Couleur selon la vitesse (vert < 50, orange 50-70, rouge > 70)
                if speed < 50:
                    color = COLORS['success']
                elif speed < 70:
                    color = COLORS['warning']
                else:
                    color = COLORS['primary']
                
                # Remplissage avec coins arrondis simulés
                if fill_width > 2:
                    cv2.rectangle(frame, (bar_x + 2, bar_y + 2),
                                (bar_x + fill_width + 2, bar_y + bar_height - 2),
                                color, -1)
                
                # Texte de vitesse sur la barre (si assez large)
                if fill_width > 20:
                    speed_text = f"{speed:.0f}"
                    text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    text_x = bar_x + 4
                    text_y = bar_y + bar_height - 3
                    cv2.putText(frame, speed_text, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['white'], 1, cv2.LINE_AA)
        
        return frame
    
    def draw_enhanced_labels(self, frame, detections, labels):
        """Dessine les labels sans bordures - texte avec contour pour meilleure lisibilité
        Les labels des véhicules trop éloignés ne sont pas affichés pour éviter le chevauchement"""
        h, w = frame.shape[:2]
        
        for idx, (box, label) in enumerate(zip(detections.xyxy, labels)):
            # Calculer la taille de la bounding box
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            box_area = box_width * box_height
            
            # Seuil minimum: ne pas afficher les labels si le véhicule est trop petit (éloigné)
            # Basé sur la taille de la bounding box - ajustable selon vos besoins
            min_box_area = 1500  # Surface minimale en pixels (peut être ajusté)
            min_box_height = 40  # Hauteur minimale en pixels
            
            # Filtrer les véhicules trop éloignés (petits)
            if box_area < min_box_area or box_height < min_box_height:
                continue  # Ne pas afficher le label pour ce véhicule
            
            # Position du label (au-dessus de la barre de vitesse)
            label_x = int(box[0])
            label_y = int(box[1]) - 60
            
            # Vérifier que le label est dans les limites de l'image
            if label_y < 0:
                continue
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Texte avec contour (shadow/outline) pour lisibilité sans bordures rectangulaires
            # Contour noir épais
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                font,
                font_scale,
                (0, 0, 0),  # Noir pour le contour
                thickness + 2,
                cv2.LINE_AA
            )
            
            # Texte principal en blanc pour excellente lisibilité
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                font,
                font_scale,
                (255, 255, 255),  # Blanc
                thickness,
                cv2.LINE_AA
            )
        
        return frame
    
    def annotate(self, frame, detections, labels, line_zone, avg_speed):
        """Fonction principale d'annotation - Design simplifié et clair"""
        # 1. Zone de transformation (en arrière-plan)
        frame = self.draw_zone_polygon(frame)
        
        # 2. Boîtes englobantes (couche principale)
        frame = self.box_annotator.annotate(frame, detections)
        
        # 3. Indicateurs de vitesse (barres - avant les labels)
        frame = self.draw_vehicle_indicators(frame, detections)
        
        # 4. Labels lisibles sans bordures (au-dessus de tout)
        frame = self.draw_enhanced_labels(frame, detections, labels)
        
        # 5. Ligne de comptage (toujours visible)
        frame = self.draw_line_zone(frame, line_zone)
        
        # 6. Panneau d'informations (superposé en dernier)
        vehicle_count = len(detections)
        frame = self.draw_overlay_panel(frame, line_zone, avg_speed, vehicle_count)
        
        return frame

# =====================================================
# INITIALISATION
# =====================================================

model = YOLO(YOLO_MODEL_PATH)
print("Model classes:", model.names)

video_info = sv.VideoInfo.from_video_path(IN_VIDEO_PATH)
print(video_info)

tracker = sv.ByteTrack(frame_rate=video_info.fps)

# Line zone
offset = 50
start = sv.Point(offset, LINE_Y)
end = sv.Point(video_info.width - offset, LINE_Y)

line_zone = sv.LineZone(
    start, end,
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

# Annotateur moderne
annotator = ModernAnnotator(
    resolution_wh=video_info.resolution_wh,
    polygon_zone=SOURCE,
    line_start=start,
    line_end=end
)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, video_info.width, video_info.height)

# =====================================================
# CONFIGURATION
# =====================================================

VEHICLE_CLASS_IDS = [3, 4, 5, 8, 9]
track_class_map = {}

vehicle_history = defaultdict(lambda: {
    'speeds': [],
    'positions': [],
    'frames': []
})

# CSV
csv_file = open('vehicle_features.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'frame', 'tracker_id', 'vehicle_class', 'speed', 'acceleration',
    'distance_to_nearest', 'direction_change', 'avg_traffic_speed', 'relative_speed'
])

# =====================================================
# FONCTIONS
# =====================================================

def calculate_acceleration(tracker_id, current_speed):
    history = vehicle_history[tracker_id]['speeds']
    if len(history) < 2:
        return 0.0
    prev_speed = history[-1]
    return round(current_speed - prev_speed, 2)

def calculate_distance_to_nearest(detections, current_idx):
    if len(detections) <= 1:
        return 999.0
    
    current_box = detections.xyxy[current_idx]
    current_center = np.array([
        (current_box[0] + current_box[2]) / 2,
        (current_box[1] + current_box[3]) / 2
    ])
    
    min_distance = 999.0
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
    history = vehicle_history[tracker_id]['positions']
    if len(history) < 2:
        return 0.0
    
    prev_pos = np.array(history[-1])
    prev_prev_pos = np.array(history[-2]) if len(history) >= 2 else prev_pos
    current_pos = np.array(current_position)
    
    vec1 = prev_pos - prev_prev_pos
    vec2 = current_pos - prev_pos
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 < 0.1 or norm2 < 0.1:
        return 0.0
    
    cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    return round(angle, 2)

def calculate_avg_traffic_speed(detections):
    if 'speed' not in detections.data or len(detections.data['speed']) == 0:
        return 0.0
    speeds = detections.data['speed']
    valid_speeds = speeds[speeds > 0]
    if len(valid_speeds) == 0:
        return 0.0
    return round(np.mean(valid_speeds), 2)

# =====================================================
# BOUCLE PRINCIPALE
# =====================================================

frame_generator = sv.get_video_frames_generator(IN_VIDEO_PATH)
frame_count = 0

with sv.VideoSink(OUT_VIDEO_PATH, video_info) as sink:
    for frame in frame_generator:
        frame_count += 1

        # Détection
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filtrage
        mask = np.isin(detections.class_id, VEHICLE_CLASS_IDS)
        detections = detections[mask]

        if len(detections) == 0:
            sink.write_frame(frame)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Tracking
        detections = tracker.update_with_detections(detections)

        # Classe stable
        class_names = []
        for tid, cid in zip(detections.tracker_id, detections.class_id):
            if tid not in track_class_map:
                track_class_map[tid] = model.names[cid]
            class_names.append(track_class_map[tid])
        detections.data["class_name"] = np.array(class_names)

        # Comptage
        line_zone.trigger(detections)

        # Vitesse
        detections = speed_estimator.update(detections)

        # Vitesse moyenne
        avg_traffic_speed = calculate_avg_traffic_speed(detections)

        # Extraction features
        for idx, (tid, cname, speed) in enumerate(zip(
            detections.tracker_id,
            detections.data["class_name"],
            detections.data["speed"]
        )):
            box = detections.xyxy[idx]
            current_position = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            
            acceleration = calculate_acceleration(tid, speed)
            distance_to_nearest = calculate_distance_to_nearest(detections, idx)
            direction_change = calculate_direction_change(tid, current_position)
            relative_speed = round(speed - avg_traffic_speed, 2) if speed > 0 else 0.0
            
            vehicle_history[tid]['speeds'].append(speed)
            vehicle_history[tid]['positions'].append(current_position)
            vehicle_history[tid]['frames'].append(frame_count)
            
            if len(vehicle_history[tid]['speeds']) > 10:
                vehicle_history[tid]['speeds'].pop(0)
                vehicle_history[tid]['positions'].pop(0)
                vehicle_history[tid]['frames'].pop(0)
            
            csv_writer.writerow([
                frame_count, tid, cname, speed, acceleration,
                distance_to_nearest, direction_change,
                avg_traffic_speed, relative_speed
            ])

        # Labels
        labels = []
        for tid, cname, speed in zip(
            detections.tracker_id,
            detections.data["class_name"],
            detections.data["speed"]
        ):
            if speed > 0:
                labels.append(f"{cname} #{tid} | {speed:.1f} km/h")
            else:
                labels.append(f"{cname} #{tid}")

        # Annotation moderne
        annotated_frame = annotator.annotate(
            frame=frame,
            detections=detections,
            labels=labels,
            line_zone=line_zone,
            avg_speed=avg_traffic_speed
        )

        sink.write_frame(annotated_frame)
        cv2.imshow(WINDOW_NAME, annotated_frame)

        if (cv2.waitKey(1) & 0xFF == ord("q") or 
            cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1):
            break

csv_file.close()
cv2.destroyAllWindows()

print("Processing complete.")
print(f"Output video: {OUT_VIDEO_PATH}")
print(f"Features saved to: vehicle_features.csv")
print(f"Total vehicles: {line_zone.in_count + line_zone.out_count}")
print(f"In: {line_zone.in_count} | Out: {line_zone.out_count}")