# Configuration File

# Define Video Paths
IN_VIDEO_PATH = "./data/vehicles3.mp4"
OUT_VIDEO_PATH = "./data/vehicles_output2.mp4"

# YOLO Model Path
# To switch to a custom model, update the path below
# YOLO_MODEL_PATH = "./models/yolov8n.pt"
YOLO_MODEL_PATH = "./models/VisDrone_YOLO_x2.pt"

# Line Zone for counting vehicles (in pixels)
LINE_Y = 480

# Perspective Transform Points
# SOURCE_POINTS: Points in the original image
# TARGET_POINTS: Corresponding points in the top-down view (real-world mapping)
# Dans config.py
SOURCE_POINTS = [
    [450, 300],    # Point 0: Haut gauche (voie gauche)
    [1270, 300],   # Point 1: Haut droit (voie droite)
    [1900, 720],   # Point 2: Bas droit (voie droite)
    [-200, 720]    # Point 3: Bas gauche (voie gauche)
]

WIDTH, HEIGHT = 25, 100  # 25m = largeur totale des 2 voies
TARGET_POINTS = [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]

# Display Window Settings
WINDOW_NAME = "Detection + Tracking + Counting + Speed Estimation"
