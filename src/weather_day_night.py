"""
Détecteur combiné: Météo (CLIP + Flux Optique) + Jour/Nuit + Mise à jour CSV
Pour CSV avec plusieurs lignes liées à UNE SEULE vidéo
"""

import cv2
import numpy as np
from collections import Counter
import torch
from PIL import Image
import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel


class WeatherDayNightDetector:
    def __init__(self):
        print("Chargement du modèle CLIP...")
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Descriptions météo
        self.weather_labels = [
            "heavy rain falling down with water droplets and wet surfaces, rainy day with rainfall and precipitation",
            "snow falling from the sky with white snowflakes, snowy winter weather with snow on the ground",
            "thick fog and mist with low visibility, foggy weather with haze and unclear view",
            "strong wind blowing trees and leaves, windy weather with branches moving and things swaying in the wind",
            "clear blue sky with bright sunshine, sunny day with good weather and high visibility"
        ]
        
        print(f"✓ Modèle chargé sur {self.device}")
        print("✓ Détection: Météo (CLIP + Flux Optique) + Luminosité (Jour/Nuit)\n")

    def detect_wind_motion(self, prev_frame, curr_frame):
        """Détecte le vent par analyse du mouvement"""
        if prev_frame is None:
            return 0.0
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        horizontal_motion = np.mean(np.abs(flow[..., 0]))
        mean_magnitude = np.mean(mag)
        
        ang_deg = ang * 180 / np.pi
        hist_ang = np.histogram(ang_deg.flatten(), bins=36, range=(0, 360))[0]
        direction_coherence = np.max(hist_ang) / (np.sum(hist_ang) + 1e-5)
        
        wind_score = (
            min(horizontal_motion / 3.0, 1.0) * 0.4 +
            min(mean_magnitude / 4.0, 1.0) * 0.4 +
            direction_coherence * 0.2
        )
        
        return wind_score

    def predict_frame(self, frame, prev_frame=None):
        """Prédire météo + calculer luminosité"""
        # === LUMINOSITÉ ===
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # === MÉTÉO ===
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        inputs = self.processor(
            text=self.weather_labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs_visual = logits_per_image.softmax(dim=1)[0].cpu().numpy()
        
        wind_motion_score = self.detect_wind_motion(prev_frame, frame)
        
        all_scores = np.zeros(5)
        all_scores[0] = probs_visual[0]  # rainy
        all_scores[1] = probs_visual[1]  # snowy
        all_scores[2] = probs_visual[2]  # foggy
        all_scores[3] = (probs_visual[3] * 0.4) + (wind_motion_score * 0.6)  # windy
        all_scores[4] = probs_visual[4]  # clear
        
        all_scores = all_scores / (np.sum(all_scores) + 1e-5)
        pred_idx = np.argmax(all_scores)
        confidence = all_scores[pred_idx]
        
        weather_names = ['rainy', 'snowy', 'foggy', 'windy', 'clear']
        
        return weather_names[pred_idx], confidence, all_scores, brightness

    def analyze_video(self, video_path, sample_rate=15, confidence_threshold=0.25, brightness_threshold=80):
        """Analyser vidéo: météo + jour/nuit"""
        if not os.path.exists(video_path):
            print(f"❌ Erreur: Le fichier {video_path} n'existe pas")
            return None, None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Erreur: Impossible d'ouvrir la vidéo")
            return None, None
        
        predictions = []
        confidences = []
        all_scores_list = []
        brightness_values = []
        frame_count = 0
        prev_frame = None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        weather_names = ['rainy', 'snowy', 'foggy', 'windy', 'clear']
        
        print("╔════════════════════════════════════════════════════════════╗")
        print("║        ANALYSE MÉTÉO + JOUR/NUIT - CLIP + OPTIQUE         ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"\n📹 Vidéo: {os.path.basename(video_path)}")
        print(f"   Frames: {total_frames} | FPS: {fps:.1f} | Durée: {total_frames/fps:.1f}s")
        print(f"   Sample: 1 frame tous les {sample_rate} frames\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                weather, conf, scores, brightness = self.predict_frame(frame, prev_frame)
                
                if conf > confidence_threshold:
                    predictions.append(weather)
                    confidences.append(conf)
                    all_scores_list.append(scores)
                    brightness_values.append(brightness)
                
                prev_frame = frame.copy()
            
            frame_count += 1
        
        cap.release()
        
        # === RÉSULTATS ===
        if predictions and brightness_values:
            # Météo
            counter = Counter(predictions)
            results = counter.most_common()
            avg_scores = np.mean(all_scores_list, axis=0)
            max_score_idx = np.argmax(avg_scores)
            final_weather = weather_names[max_score_idx]
            
            if avg_scores[3] > 0.30:
                final_weather = 'windy'
            
            # Jour/Nuit
            avg_brightness = np.mean(brightness_values)
            day_night = "night" if avg_brightness < brightness_threshold else "day"
            
            # Affichage
            print("\n" + "="*70)
            print("📊 RÉSULTATS DE L'ANALYSE")
            print("="*70)
            print(f"Frames analysées: {len(predictions)}")
            print(f"Confiance moyenne: {np.mean(confidences):.3f}")
            print(f"Luminosité moyenne: {avg_brightness:.1f}\n")
            
            print("Scores moyens par condition météo:")
            for i, (name, score) in enumerate(zip(weather_names, avg_scores)):
                bar = "█" * int(score * 50)
                indicator = " ⭐" if i == max_score_idx else ""
                print(f"  {name:10s}: {score:.3f} {bar}{indicator}")
            
            print("\nDétections par votes:")
            for weather, count in results:
                percentage = (count / len(predictions)) * 100
                bar = "█" * int(percentage / 2)
                print(f"  {weather:10s}: {count:4d} ({percentage:5.1f}%) {bar}")
            
            print("="*70)
            
            weather_emoji = {'rainy': '🌧️', 'snowy': '❄️', 'foggy': '🌫️', 'windy': '💨', 'clear': '☀️'}
            daynight_emoji = {'day': '☀️', 'night': '🌙'}
            
            print(f"\n🎯 MÉTÉO: {weather_emoji.get(final_weather, '🌤️')} {final_weather.upper()}")
            print(f"🎯 PÉRIODE: {daynight_emoji[day_night]} {day_night.upper()} (luminosité: {avg_brightness:.1f})")
            print("="*70 + "\n")
            
            return final_weather, day_night
        
        return None, None


def add_weather_to_csv(csv_path, video_path, output_csv=None):
    """
    Ajouter les colonnes 'weather' et 'day_night' à un CSV lié à UNE SEULE vidéo
    Toutes les lignes du CSV auront la même valeur pour ces colonnes
    
    Args:
        csv_path: Chemin du fichier CSV (ex: vehicle_features_labeled.csv)
        video_path: Chemin de la vidéo correspondante
        output_csv: Chemin du CSV de sortie (si None, écrase l'original)
    """
    # Vérifier l'existence des fichiers
    if not os.path.exists(csv_path):
        print(f"❌ Erreur: Le fichier CSV {csv_path} n'existe pas")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ Erreur: Le fichier vidéo {video_path} n'existe pas")
        return
    
    # Charger le CSV
    print(f"\n📊 Chargement du CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Lignes: {len(df)} | Colonnes: {list(df.columns)}\n")
    
    # Analyser la vidéo
    detector = WeatherDayNightDetector()
    weather, day_night = detector.analyze_video(video_path)
    
    if weather and day_night:
        # Ajouter les colonnes (même valeur pour toutes les lignes)
        df['weather'] = weather
        df['day_night'] = day_night
        
        # Sauvegarder
        output_path = output_csv if output_csv else csv_path
        df.to_csv(output_path, index=False)
        
        print(f"\n{'='*70}")
        print(f"✅ CSV mis à jour: {output_path}")
        print(f"{'='*70}")
        print(f"   Météo détectée: {weather}")
        print(f"   Période: {day_night}")
        print(f"   Appliqué à {len(df)} lignes\n")
        
        # Afficher un aperçu
        print("Aperçu des premières lignes:")
        print(df[['frame', 'tracker_id', 'vehicle_class', 'weather', 'day_night']].head(10))
        
        print(f"\n📈 Statistiques finales:")
        print(f"   Colonnes totales: {len(df.columns)}")
        print(f"   Nouvelles colonnes: weather, day_night")
        
    else:
        print("\n❌ Impossible d'analyser la vidéo. CSV non modifié.")


# ============================================================================
# UTILISATION
# ============================================================================

if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════╗")
    print("║    AJOUT MÉTÉO + JOUR/NUIT À UN CSV LIÉ À UNE VIDÉO       ║")
    print("╚════════════════════════════════════════════════════════════╝\n")
    
    # Configuration
    csv_path = "vehicle_features_labeled.csv"  # Ton CSV avec 22544 lignes
    video_path = "/data/vehicles.mp4"      # La vidéo correspondante
    output_csv = "vehicle_features_labeled_updated.csv"  # Sortie (ou None)
    
    # Lancer l'analyse
    add_weather_to_csv(
        csv_path=csv_path,
        video_path=video_path,
        output_csv=output_csv  # Mettre None pour écraser le fichier original
    )
    
    print("\n✅ Traitement terminé!")