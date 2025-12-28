# ====================================================================
# train_random_forest.py - Entraînement Random Forest
# ====================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class RandomForestTrainer:
    """Entraînement Random Forest"""
    
    def __init__(self, csv_path='vehicle_features_labeled.csv'):
        self.csv_path = csv_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        self.feature_names = None
        
        Path('models').mkdir(exist_ok=True)
        
    def load_data(self):
        """Charge données"""
        
        print("="*60)
        print("📊 CHARGEMENT DONNÉES - RANDOM FOREST")
        print("="*60)
        print()
        
        self.df = pd.read_csv(self.csv_path)
        
        print(f"✅ Données chargées : {len(self.df)} véhicules")
        print()
        
        if 'danger_label' not in self.df.columns:
            raise ValueError("Colonne 'danger_label' non trouvée")
        
        label_counts = self.df['danger_label'].value_counts().sort_index()
        for label, count in label_counts.items():
            label_name = ['Normal', 'Suspect', 'Dangereux'][int(label)]
            print(f"  {label_name}: {count} ({count/len(self.df)*100:.1f}%)")
        
        print()
        
        return self.df
    
    def prepare_data(self):
        """Prépare données"""
        
        print("="*60)
        print("🔧 PRÉPARATION DONNÉES - RANDOM FOREST")
        print("="*60)
        print()
        
        feature_cols = [
            'speed', 'relative_speed', 'acceleration',
            'distance_to_nearest', 'direction_change'
        ]
        
        self.feature_names = feature_cols
        
        missing_cols = [col for col in feature_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes : {missing_cols}")
        
        X = self.df[feature_cols].copy()
        X = X.fillna(0)
        y = self.df['danger_label'].values
        
        print(f"Features : {len(feature_cols)}")
        for col in feature_cols:
            print(f"  • {col}")
        print()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set : {len(self.X_train)} échantillons")
        print(f"Test set : {len(self.X_test)} échantillons")
        print()
        
        # Random Forest ne nécessite pas forcément normalisation
        # mais on peut l'ajouter pour cohérence
        print("Normalisation (optionnelle pour RF)...")
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("✅ Données préparées")
        print()
        
    def train_random_forest(self, optimize=True):
        """Entraîne Random Forest"""
        
        print("="*60)
        print("🌲 ENTRAÎNEMENT RANDOM FOREST")
        print("="*60)
        print()
        
        if optimize:
            print("Mode : Optimisation Grid Search")
            print("⚠️  Cela peut prendre 5-10 minutes...")
            print()
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            print("\n✅ Grid Search terminé")
            print()
            print("🏆 MEILLEURS PARAMÈTRES :")
            for param, value in grid_search.best_params_.items():
                print(f"  {param} : {value}")
            print(f"  Score CV : {grid_search.best_score_:.4f}")
            print()
            
            self.model = grid_search.best_estimator_
        
        else:
            print("Mode : Configuration par défaut")
            print()
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(self.X_train, self.y_train)
            
            print("✅ Entraînement terminé")
            print()
        
    def evaluate_model(self):
        """Évalue modèle"""
        
        print("="*60)
        print("📊 ÉVALUATION RANDOM FOREST")
        print("="*60)
        print()
        
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"🎯 Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
        print()
        
        print("📋 RAPPORT DE CLASSIFICATION :")
        print()
        target_names = ['Normal', 'Suspect', 'Dangereux']
        print(classification_report(self.y_test, y_pred, target_names=target_names, digits=4))
        
        print("🔢 MATRICE DE CONFUSION :")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print()
        
        self._plot_confusion_matrix(cm, target_names)
        
        # Feature Importance (IMPORTANT pour Random Forest)
        self._plot_feature_importance()
        
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        print(f"🔄 Cross-Validation (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print()
        
        return accuracy
        
    def _plot_confusion_matrix(self, cm, target_names):
        """Visualise matrice confusion"""
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Matrice de Confusion - Random Forest', fontsize=14, fontweight='bold')
        plt.ylabel('Classe Réelle')
        plt.xlabel('Classe Prédite')
        plt.tight_layout()
        plt.savefig('rf_confusion_matrix.png', dpi=150)
        print(f"✅ Graphique : rf_confusion_matrix.png")
        plt.close()
        print()
    
    def _plot_feature_importance(self):
        """Visualise importance des features"""
        
        print("🔍 FEATURE IMPORTANCE :")
        print()
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.to_string(index=False))
        print()
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'], color='forestgreen')
        plt.xlabel('Importance')
        plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('rf_feature_importance.png', dpi=150)
        print(f"✅ Graphique : rf_feature_importance.png")
        plt.close()
        print()
    
    def save_model(self):
        """Sauvegarde modèle"""
        
        print("="*60)
        print("💾 SAUVEGARDE MODÈLE RANDOM FOREST")
        print("="*60)
        print()
        
        joblib.dump(self.model, 'models/rf_model.pkl')
        joblib.dump(self.scaler, 'models/rf_scaler.pkl')
        
        print("✅ Modèle RF : models/rf_model.pkl")
        print("✅ Scaler : models/rf_scaler.pkl")
        print()
        
    def run_full_training(self, optimize=True):
        """Pipeline complet"""
        
        print("\n")
        print("="*60)
        print("🚀 ENTRAÎNEMENT RANDOM FOREST - PIPELINE COMPLET")
        print("="*60)
        print()
        
        self.load_data()
        self.prepare_data()
        self.train_random_forest(optimize=optimize)
        accuracy = self.evaluate_model()
        self.save_model()
        
        print("="*60)
        print("✅ RANDOM FOREST TERMINÉ")
        print("="*60)
        print(f"Accuracy finale : {accuracy:.4f}")
        print()
        
        return self.model, accuracy


if __name__ == "__main__":
    
    if not Path('vehicle_features_labeled.csv').exists():
        print("❌ Fichier vehicle_features_labeled.csv non trouvé")
        exit(1)
    
    trainer = RandomForestTrainer('vehicle_features_labeled.csv')
    model, acc = trainer.run_full_training(optimize=True)