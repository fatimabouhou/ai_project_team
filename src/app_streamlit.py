# ====================================================================
# app_streamlit.py - Interface Streamlit Professionnelle
# ====================================================================

"""
Interface Streamlit pour visualiser les résultats ML
Design moderne avec couleurs douces et pastel
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# TensorFlow pour LSTM
try:
    from tensorflow import keras
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False

# ====================================================================
# CONFIGURATION PAGE
# ====================================================================

st.set_page_config(
    page_title="Traffic AI Classifier",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================================
# PALETTE DE COULEURS DOUCES
# ====================================================================

COLORS = {
    # Couleurs nudes et pastel pour graphiques
    'normal': '#A8D5BA',  # Vert doux
    'normal_dark': '#7DBFA0',  # Vert plus foncé pour contraste
    'suspect': '#FFD8A8',  # Orange doux
    'suspect_dark': '#FFC085',  # Orange plus foncé
    'danger': '#FFB3B3',  # Rouge doux
    'danger_dark': '#FF9999',  # Rouge plus foncé
    
    # Dégradés pour accuracy
    'gradient_start': '#E8E8E8',  # Gris très clair
    'gradient_end': '#B0B0B0',    # Gris moyen
    'accent_light': '#D4E6F1',    # Bleu très clair
    'accent_medium': '#A9CCE3',   # Bleu clair
    'accent_dark': '#7FB3D5',     # Bleu moyen
    
    # Couleurs neutres
    'neutral_light': '#F5F5F5',
    'neutral_medium': '#E0E0E0',
    'neutral_dark': '#9E9E9E',
    'text_dark': '#424242',
    'text_light': '#757575',
    
    # Couleurs d'arrière-plan
    'bg_light': '#FAFAFA',
    'bg_white': '#FFFFFF',
    'bg_card': '#FFFFFF',
    
    # Couleurs pour cartes de classes
    'card_normal': '#E8F5E9',     # Vert très clair
    'card_suspect': '#FFF3E0',    # Orange très clair
    'card_danger': '#FFEBEE',     # Rouge très clair
    'card_normal_border': '#C8E6C9',  # Bordure verte
    'card_suspect_border': '#FFE0B2', # Bordure orange
    'card_danger_border': '#FFCDD2',  # Bordure rouge
}

# ====================================================================
# CSS PERSONNALISÉ (Style Moderne avec couleurs douces)
# ====================================================================

st.markdown(f"""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    /* Global */
    html, body, [class*="css"] {{
        font-family: 'Inter', 'Source Sans Pro', sans-serif;
        background-color: {COLORS['bg_light']};
    }}
    
    /* Header */
    .main-header {{
        background: linear-gradient(135deg, {COLORS['suspect_dark']} 0%, {COLORS['neutral_dark']} 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: none;
    }}
    
    .main-header h1 {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        letter-spacing: -0.5px;
        color: #ffffff;
    }}
    
    .main-header p {{
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
        color: #e0e0e0;
    }}
    
    /* Metrics Cards - Style minimaliste */
    .metric-card {{
        background: {COLORS['bg_card']};
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        border: 1px solid {COLORS['neutral_medium']};
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }}
    
    .metric-card:hover {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        transform: translateY(-1px);
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['text_dark']};
        margin: 0;
    }}
    
    .metric-label {{
        font-size: 0.875rem;
        color: {COLORS['text_light']};
        margin-top: 0.5rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }}
    
    /* Buttons - Style discret */
    .stButton>button {{
        background: linear-gradient(135deg, {COLORS['suspect_dark']} 0%, {COLORS['neutral_dark']} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        letter-spacing: 0.3px;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background: {COLORS['neutral_light']};
        border-right: 1px solid {COLORS['neutral_medium']};
    }}
    
    /* Tables */
    .dataframe {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 8px rgba(0,0,0,0.04);
        border: 1px solid {COLORS['neutral_medium']};
        background: {COLORS['bg_white']};
    }}
    
    /* Status boxes - Style très léger */
    .success-box {{
        background: {COLORS['card_normal']};
        border-left: 3px solid {COLORS['normal_dark']};
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
    }}
    
    .warning-box {{
        background: {COLORS['card_suspect']};
        border-left: 3px solid {COLORS['suspect_dark']};
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
    }}
    
    .error-box {{
        background: {COLORS['card_danger']};
        border-left: 3px solid {COLORS['danger_dark']};
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
    }}
    
    .info-box {{
        background: {COLORS['accent_light']};
        border-left: 3px solid {COLORS['accent_dark']};
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {COLORS['neutral_light']};
        padding: 8px;
        border-radius: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        border-radius: 8px;
        padding: 10px 20px;
        background-color: {COLORS['bg_white']};
        border: 1px solid {COLORS['neutral_medium']};
        transition: all 0.3s ease;
        color: {COLORS['text_dark']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['text_dark']};
        color: white;
        border-color: {COLORS['text_dark']};
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    
    /* Custom dividers */
    .custom-divider {{
        height: 1px;
        background: linear-gradient(to right, transparent, {COLORS['neutral_medium']}, transparent);
        margin: 2rem 0;
    }}
    
    /* Progress bars */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
    }}
    
    /* Cartes spéciales pour les classes */
    .class-card-normal {{
        background: {COLORS['card_normal']};
        border: 1px solid {COLORS['card_normal_border']};
        border-radius: 10px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }}
    
    .class-card-suspect {{
        background: {COLORS['card_suspect']};
        border: 1px solid {COLORS['card_suspect_border']};
        border-radius: 10px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }}
    
    .class-card-danger {{
        background: {COLORS['card_danger']};
        border: 1px solid {COLORS['card_danger_border']};
        border-radius: 10px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }}
    
    .class-card-normal:hover, .class-card-suspect:hover, .class-card-danger:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }}
    
    /* Badges subtils */
    .normal-badge {{
        background-color: {COLORS['card_normal']};
        color: {COLORS['normal_dark']};
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid {COLORS['card_normal_border']};
    }}
    
    .suspect-badge {{
        background-color: {COLORS['card_suspect']};
        color: {COLORS['suspect_dark']};
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid {COLORS['card_suspect_border']};
    }}
    
    .danger-badge {{
        background-color: {COLORS['card_danger']};
        color: {COLORS['danger_dark']};
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid {COLORS['card_danger_border']};
    }}
    
    /* Style pour les graphiques */
    .chart-container {{
        background: {COLORS['bg_white']};
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid {COLORS['neutral_medium']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    
</style>
""", unsafe_allow_html=True)

# ====================================================================
# CONFIGURATION
# ====================================================================

class Config:
    """Configuration robuste des chemins"""

    BASE_DIR = Path(__file__).resolve().parents[1]  # Racine du projet

    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUTS_DIR = BASE_DIR / "outputs"

    VIDEO_INPUT = DATA_DIR / "vehicles.mp4"
    VIDEO_OUTPUT = DATA_DIR / "vehicles_output.mp4"

    CSV_FEATURES = BASE_DIR / "vehicle_features_labeled_updated.csv"

    MODELS = {
        "SVM": ("svm_model.pkl", "svm_scaler.pkl"),
        "Random Forest": ("rf_model.pkl", "rf_scaler.pkl"),
        "XGBoost": ("xgboost_model.pkl", None),
        "KNN": ("knn_model.pkl", "knn_scaler.pkl"),
        "Decision Tree": ("decision_tree_model.pkl", None),
        "LSTM": ("lstm_model.h5", "lstm_scaler.pkl"),
        "MLP": ("mlp_model.h5", None),
    }


config = Config()

# ====================================================================
# GESTIONNAIRE MODÈLES
# ====================================================================

@st.cache_resource
def load_models():
    """Charge tous les modèles (avec cache)"""
    
    models = {}
    scalers = {}
    loaded_models = []
    failed_models = []
    
    for model_name, (model_file, scaler_file) in config.MODELS.items():
        try:
            model_path = config.MODELS_DIR / model_file
            
            if not model_path.exists():
                failed_models.append(f"{model_name}: fichier modèle non trouvé")
                continue
            
            # Charger modèle
            if model_file.endswith('.h5'):
                if not LSTM_AVAILABLE:
                    failed_models.append(f"{model_name}: TensorFlow non disponible")
                    continue
                try:
                    models[model_name] = keras.models.load_model(model_path)
                except Exception as e:
                    failed_models.append(f"{model_name}: erreur de chargement Keras - {str(e)[:100]}")
                    continue
            else:
                try:
                    models[model_name] = joblib.load(model_path)
                except Exception as e:
                    failed_models.append(f"{model_name}: erreur de chargement - {str(e)[:100]}")
                    continue
            
            # Charger scaler
            if scaler_file:
                scaler_path = config.MODELS_DIR / scaler_file
                if scaler_path.exists():
                    try:
                        scalers[model_name] = joblib.load(scaler_path)
                    except Exception as e:
                        failed_models.append(f"{model_name}: erreur chargement scaler - {str(e)[:100]}")
                else:
                    failed_models.append(f"{model_name}: fichier scaler non trouvé")
            
            loaded_models.append(model_name)
            
        except Exception as e:
            failed_models.append(f"{model_name}: erreur - {str(e)[:100]}")
    
    return models, scalers, loaded_models, failed_models

# ====================================================================
# FONCTIONS UTILITAIRES
# ====================================================================

def safe_predict(model, X, model_name):
    """Prédiction sécurisée avec gestion des erreurs"""
    try:
        if model_name in ['LSTM', 'MLP']:
            if model_name == 'LSTM':
                # Pour LSTM, reshape en 3D
                X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
                predictions = model.predict(X_reshaped, verbose=0)
            else:
                predictions = model.predict(X, verbose=0)
            
            # Pour les modèles de classification Keras
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                predictions_classes = np.argmax(predictions, axis=1)
                predictions_proba = predictions
            else:
                predictions_classes = (predictions > 0.5).astype(int)
                predictions_proba = predictions
                
        else:
            # Pour les modèles scikit-learn
            predictions_classes = model.predict(X)
            
            # Essayer d'obtenir les probabilités
            try:
                predictions_proba = model.predict_proba(X)
            except AttributeError:
                # Si predict_proba n'existe pas, créer des probabilités factices
                predictions_proba = np.zeros((len(X), 3))
                for i, pred in enumerate(predictions_classes):
                    predictions_proba[i, int(pred)] = 1.0
        
        return predictions_classes, predictions_proba
        
    except Exception as e:
        st.markdown(f'<div class="warning-box">Erreur dans {model_name}: {str(e)[:200]}</div>', unsafe_allow_html=True)
        return None, None

def predict_dataframe(models, scalers, df):
    """Prédictions sur DataFrame avec gestion robuste des erreurs"""
    
    feature_names = [
        'speed', 'relative_speed', 'avg_traffic_speed',
        'acceleration', 'distance_to_nearest', 'direction_change'
    ]
    
    results = {}
    
    for model_name, model in models.items():
        try:
            scaler = scalers.get(model_name)
            
            # Vérifier que toutes les features sont présentes
            missing_features = [f for f in feature_names if f not in df.columns]
            if missing_features:
                st.markdown(f'<div class="warning-box">Features manquantes pour {model_name}: {missing_features}</div>', unsafe_allow_html=True)
                results[model_name] = None
                continue
            
            X = df[feature_names].values
            
            # Appliquer le scaler si disponible
            if scaler:
                X = scaler.transform(X)
            
            # Faire la prédiction
            predictions, probabilities = safe_predict(model, X, model_name)
            
            if predictions is not None:
                results[model_name] = {
                    'predictions': predictions,
                    'probabilities': probabilities
                }
            else:
                results[model_name] = None
                
        except Exception as e:
            st.markdown(f'<div class="warning-box">Erreur lors de la prédiction avec {model_name}: {str(e)[:200]}</div>', unsafe_allow_html=True)
            results[model_name] = None
    
    return results

# ====================================================================
# CHARGEMENT INITIAL
# ====================================================================

# Charger les modèles avec cache
models, scalers, loaded_models, failed_models = load_models()

# ====================================================================
# HEADER
# ====================================================================

st.markdown("""
<div class="main-header">
    <h1>Traffic Behavior AI Classifier</h1>
    <p>Analyse intelligente des comportements routiers par Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ====================================================================
# SIDEBAR
# ====================================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Sélectionnez une page",
    ["Vidéos", "Dataset", "Prédictions", "Performances", "Documentation"]
)

st.sidebar.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.sidebar.subheader("Statistiques")
st.sidebar.markdown(f"**Modèles chargés :** {len(loaded_models)}")

if loaded_models:
    st.sidebar.markdown("**Modèles disponibles :**")
    for model_name in loaded_models:
        st.sidebar.markdown(f"• {model_name}")

if failed_models:
    st.sidebar.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("**Modèles non chargés :**")
    for failed in failed_models[:3]:
        st.sidebar.markdown(f"• {failed.split(':')[0]}")


# ====================================================================
# PAGE 1 : VIDÉOS
# ====================================================================

if page == "Vidéos":
    st.header("Analyse Vidéo")
    st.markdown("Visualisez les vidéos originales et annotées avec détections en temps réel")
    
    if st.button("Charger les Vidéos", key="load_videos"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vidéo Originale")
            
            if config.VIDEO_INPUT.exists():
                try:
                    st.video(str(config.VIDEO_INPUT))
                    st.markdown('<div class="success-box">Vidéo originale chargée avec succès</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-box">Erreur de chargement vidéo: {str(e)[:100]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="error-box">Vidéo introuvable: {config.VIDEO_INPUT}</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("Vidéo Annotée")
            
            if config.VIDEO_OUTPUT.exists():
                try:
                    st.video(str(config.VIDEO_OUTPUT))
                    st.markdown('<div class="success-box">Vidéo annotée chargée avec succès</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-box">Erreur de chargement vidéo: {str(e)[:100]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="error-box">Vidéo introuvable: {config.VIDEO_OUTPUT}</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>Informations :</strong><br>
            • <strong>Input :</strong> Vidéo originale du trafic routier<br>
            • <strong>Output :</strong> Vidéo annotée avec détections et tracking des véhicules
        </div>
        """, unsafe_allow_html=True)

# ====================================================================
# PAGE 2 : DATASET
# ====================================================================

elif page == "Dataset":
    st.header("Dataset Features")
    st.markdown("Exploration du dataset avec visualisations interactives")
    
    if st.button("Charger le Dataset", key="load_csv"):
        
        if not config.CSV_FEATURES.exists():
            st.markdown(f'<div class="error-box">Fichier introuvable: {config.CSV_FEATURES}</div>', unsafe_allow_html=True)
        else:
            try:
                df = pd.read_csv(config.CSV_FEATURES)
                
                # Métriques
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{len(df):,}</p>
                        <p class="metric-label">Véhicules</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{len(df.columns)}</p>
                        <p class="metric-label">Features</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    n_classes = df['danger_label'].nunique() if 'danger_label' in df.columns else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{n_classes}</p>
                        <p class="metric-label">Classes</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Distribution
                if 'danger_label' in df.columns:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.subheader("Distribution des Labels")
                    
                    label_counts = df['danger_label'].value_counts().sort_index()
                    labels = ['Normal', 'Suspect', 'Dangereux']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Graphique barres avec couleurs douces
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = [COLORS['normal'], COLORS['suspect'], COLORS['danger']]
                        
                        bars = ax.bar(labels, label_counts.values, color=colors, alpha=0.9, 
                                     edgecolor='white', linewidth=2)
                        ax.set_ylabel('Nombre de véhicules', fontweight=500, fontsize=11, color=COLORS['text_dark'])
                        ax.set_title('Distribution des Labels', fontweight=600, fontsize=13, pad=20, color=COLORS['text_dark'])
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color(COLORS['neutral_medium'])
                        ax.spines['bottom'].set_color(COLORS['neutral_medium'])
                        ax.tick_params(colors=COLORS['text_light'])
                        ax.grid(axis='y', alpha=0.1, linestyle='-', color=COLORS['neutral_medium'])
                        
                        # Ajouter les valeurs sur les barres avec une couleur douce
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height):,}', ha='center', va='bottom', 
                                   fontweight=600, fontsize=10, color=COLORS['text_dark'])
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        # Graphique vitesse avec couleurs douces
                        if 'speed' in df.columns:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(df['speed'], bins=30, color=COLORS['accent_medium'], 
                                   alpha=0.7, edgecolor='white', linewidth=1.5)
                            ax.set_xlabel('Vitesse (km/h)', fontweight=500, fontsize=11, color=COLORS['text_dark'])
                            ax.set_ylabel('Fréquence', fontweight=500, fontsize=11, color=COLORS['text_dark'])
                            ax.set_title('Distribution Vitesse', fontweight=600, fontsize=13, pad=20, color=COLORS['text_dark'])
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_color(COLORS['neutral_medium'])
                            ax.spines['bottom'].set_color(COLORS['neutral_medium'])
                            ax.tick_params(colors=COLORS['text_light'])
                            ax.grid(axis='y', alpha=0.1, linestyle='-', color=COLORS['neutral_medium'])
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Tableau
                st.subheader("Aperçu des Données")
                st.dataframe(df.head(100), use_container_width=True)
                
                # Informations détaillées
                with st.expander("Statistiques détaillées"):
                    st.write("**Types de données :**")
                    st.write(df.dtypes)
                    
                    st.write("**Valeurs manquantes :**")
                    missing = df.isnull().sum()
                    st.write(missing[missing > 0])
                    
                    if 'danger_label' in df.columns:
                        st.write("**Distribution des labels :**")
                        for label, count in label_counts.items():
                            label_name = ['Normal', 'Suspect', 'Dangereux'][int(label)]
                            percentage = (count / len(df)) * 100
                            st.write(f"• {label_name}: {count} ({percentage:.1f}%)")
                
                st.markdown(f'<div class="success-box">Dataset chargé avec succès : {len(df)} véhicules analysés</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">Erreur lors du chargement du CSV: {str(e)}</div>', unsafe_allow_html=True)

# ====================================================================
# PAGE 3 : PRÉDICTIONS
# ====================================================================

elif page == "Prédictions":
    st.header("Prédictions Multi-Modèles")
    st.markdown("Résultats de classification par différents algorithmes de Machine Learning")
    
    if not models:
        st.markdown('<div class="error-box">Aucun modèle chargé. Vérifiez les fichiers dans le dossier "models".</div>', unsafe_allow_html=True)
    elif st.button("Générer Prédictions", key="predict"):
        
        if not config.CSV_FEATURES.exists():
            st.markdown(f'<div class="error-box">CSV introuvable: {config.CSV_FEATURES}</div>', unsafe_allow_html=True)
        else:
            try:
                df = pd.read_csv(config.CSV_FEATURES)
                
                with st.spinner("Génération des prédictions en cours..."):
                    results = predict_dataframe(models, scalers, df)
                
                # Filtrer les résultats non nuls
                valid_results = {k: v for k, v in results.items() if v is not None}
                
                if not valid_results:
                    st.markdown('<div class="error-box">Aucune prédiction valide générée. Vérifiez les modèles et les données.</div>', unsafe_allow_html=True)
                else:
                    # Rapport par modèle
                    st.subheader("Résultats par Modèle")
                    
                    stats_data = []
                    
                    for model_name, result in valid_results.items():
                        predictions = result['predictions']
                        
                        # Compter les prédictions par classe
                        if predictions.ndim == 1:
                            normal_count = (predictions == 0).sum()
                            suspect_count = (predictions == 1).sum()
                            danger_count = (predictions == 2).sum()
                        else:
                            predictions_classes = np.argmax(predictions, axis=1)
                            normal_count = (predictions_classes == 0).sum()
                            suspect_count = (predictions_classes == 1).sum()
                            danger_count = (predictions_classes == 2).sum()
                        
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{model_name}**")
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 3px solid {COLORS['normal_dark']};">
                                <p class="metric-value">{normal_count}</p>
                                <p class="metric-label">Normal</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 3px solid {COLORS['suspect_dark']};">
                                <p class="metric-value">{suspect_count}</p>
                                <p class="metric-label">Suspect</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 3px solid {COLORS['danger_dark']};">
                                <p class="metric-value">{danger_count}</p>
                                <p class="metric-label">Dangereux</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        stats_data.append({
                            'Modèle': model_name,
                            'Normal': int(normal_count),
                            'Suspect': int(suspect_count),
                            'Dangereux': int(danger_count)
                        })
                    
                    # Graphiques comparatifs
                    if stats_data:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.subheader("Comparaison Visuelle")
                        
                        stats_df = pd.DataFrame(stats_data)
                        
                        fig, ax = plt.subplots(figsize=(14, 7))
                        
                        x = np.arange(len(stats_df))
                        width = 0.25
                        
                        # Barres avec couleurs douces
                        ax.bar(x - width, stats_df['Normal'], width, 
                              label='Normal', color=COLORS['normal'], alpha=0.9, edgecolor='white', linewidth=2)
                        ax.bar(x, stats_df['Suspect'], width,
                              label='Suspect', color=COLORS['suspect'], alpha=0.9, edgecolor='white', linewidth=2)
                        ax.bar(x + width, stats_df['Dangereux'], width,
                              label='Dangereux', color=COLORS['danger'], alpha=0.9, edgecolor='white', linewidth=2)
                        
                        ax.set_xlabel('Modèles', fontweight=500, fontsize=12, color=COLORS['text_dark'])
                        ax.set_ylabel('Nombre de prédictions', fontweight=500, fontsize=12, color=COLORS['text_dark'])
                        ax.set_title('Comparaison des Prédictions', fontweight=600, fontsize=14, pad=20, color=COLORS['text_dark'])
                        ax.set_xticks(x)
                        ax.set_xticklabels(stats_df['Modèle'], rotation=45, ha='right', color=COLORS['text_dark'])
                        ax.legend(frameon=False, fontsize=11)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color(COLORS['neutral_medium'])
                        ax.spines['bottom'].set_color(COLORS['neutral_medium'])
                        ax.tick_params(colors=COLORS['text_light'])
                        ax.grid(axis='y', alpha=0.1, linestyle='-', color=COLORS['neutral_medium'])
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tableau résultats
                    st.subheader("Tableau Détaillé")
                    
                    results_df = df.copy()
                    
                    for model_name, result in valid_results.items():
                        predictions = result['predictions']
                        
                        if predictions.ndim == 1:
                            predictions_classes = predictions
                        else:
                            predictions_classes = np.argmax(predictions, axis=1)
                        
                        results_df[f'{model_name}_label'] = [
                            ['Normal', 'Suspect', 'Dangereux'][int(p)] 
                            for p in predictions_classes
                        ]
                    
                    # Sauvegarder
                    try:
                        output_path = 'predictions_all_models.csv'
                        results_df.to_csv(output_path, index=False)
                        st.markdown(f'<div class="success-box">Résultats sauvegardés : {output_path}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="warning-box">Impossible de sauvegarder: {str(e)}</div>', unsafe_allow_html=True)
                    
                    # Afficher un aperçu
                    preview_cols = ['speed', 'acceleration', 'distance_to_nearest']
                    preview_cols += [c for c in results_df.columns if '_label' in c]
                    
                    st.dataframe(results_df[preview_cols].head(100), use_container_width=True)
                    
            except Exception as e:
                st.markdown(f'<div class="error-box">Erreur lors des prédictions: {str(e)}</div>', unsafe_allow_html=True)

# ====================================================================
# PAGE 4 : PERFORMANCES
# ====================================================================

elif page == "Performances":
    st.header("Comparaison des Performances")
    st.markdown("Classement par accuracy sur le dataset de test")
    
    if not models:
        st.markdown('<div class="error-box">Aucun modèle chargé. Vérifiez les fichiers dans le dossier "models".</div>', unsafe_allow_html=True)
    elif st.button("Calculer Performances", key="compare"):
        
        if not config.CSV_FEATURES.exists():
            st.markdown('<div class="error-box">CSV introuvable</div>', unsafe_allow_html=True)
        else:
            try:
                df = pd.read_csv(config.CSV_FEATURES)
                
                if 'danger_label' not in df.columns:
                    st.markdown('<div class="error-box">Pas de labels de vérité terrain dans le dataset</div>', unsafe_allow_html=True)
                else:
                    with st.spinner("Calcul des performances en cours..."):
                        results = predict_dataframe(models, scalers, df)
                    
                    # Filtrer les résultats valides
                    valid_results = {k: v for k, v in results.items() if v is not None}
                    
                    if not valid_results:
                        st.markdown('<div class="error-box">Aucun résultat valide pour calculer les performances.</div>', unsafe_allow_html=True)
                    else:
                        y_true = df['danger_label'].values
                        
                        accuracies = {}
                        
                        for model_name, result in valid_results.items():
                            predictions = result['predictions']
                            
                            if predictions.ndim == 1:
                                y_pred = predictions
                            else:
                                y_pred = np.argmax(predictions, axis=1)
                            
                            # Calculer l'accuracy
                            try:
                                accuracy = (y_pred == y_true).mean() * 100
                                accuracies[model_name] = accuracy
                            except Exception as e:
                                st.markdown(f'<div class="warning-box">Impossible de calculer l\'accuracy pour {model_name}: {str(e)}</div>', unsafe_allow_html=True)
                        
                        if accuracies:
                            # Classement
                            sorted_acc = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
                            
                            st.subheader("Classement des Modèles")
                            
                            for i, (model, acc) in enumerate(sorted_acc):
                                rank = f"{i+1}."
                                
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**{rank} {model}**")
                                
                                with col2:
                                    # Dégradé de couleurs selon le classement
                                    if i == 0:
                                        border_color = COLORS['normal_dark']
                                    elif i == 1:
                                        border_color = COLORS['suspect_dark']
                                    elif i == 2:
                                        border_color = COLORS['danger_dark']
                                    else:
                                        border_color = COLORS['neutral_medium']
                                    
                                    st.markdown(f"""
                                    <div class="metric-card" style="border-left: 3px solid {border_color};">
                                        <p class="metric-value">{acc:.2f}%</p>
                                        <p class="metric-label">Accuracy</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Graphique avec dégradé doux
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.subheader("Visualisation")
                            
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            models_list = [m for m, _ in sorted_acc]
                            accs_list = [a for _, a in sorted_acc]
                            
                            # Dégradé doux du clair au foncé
                            colors_gradient = plt.cm.Greys(np.linspace(0.2, 0.6, len(models_list)))
                            
                            bars = ax.barh(models_list, accs_list, color=colors_gradient, alpha=0.85, 
                                          edgecolor='white', linewidth=2)
                            
                            ax.set_xlabel('Accuracy (%)', fontweight=500, fontsize=12, color=COLORS['text_dark'])
                            ax.set_title('Classement des Modèles', fontweight=600, fontsize=14, pad=20, color=COLORS['text_dark'])
                            ax.set_xlim(0, 100)
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            ax.spines['bottom'].set_color(COLORS['neutral_medium'])
                            ax.tick_params(left=False, colors=COLORS['text_light'])
                            ax.grid(axis='x', alpha=0.1, linestyle='-', color=COLORS['neutral_medium'])
                            
                            # Ajouter les valeurs avec une couleur discrète
                            for i, (bar, acc) in enumerate(zip(bars, accs_list)):
                                ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2,
                                       f'{acc:.2f}%', va='center', fontweight=600, 
                                       fontsize=11, color=COLORS['text_dark'])
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-box">Aucune accuracy calculée.</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">Erreur lors du calcul des performances: {str(e)}</div>', unsafe_allow_html=True)

# ====================================================================
# PAGE 5 : DOCUMENTATION
# ====================================================================

elif page == "Documentation":
    st.header("Documentation Technique")
    
    # Statistiques de chargement
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(loaded_models)}</p>
            <p class="metric-label">Modèles chargés</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if failed_models:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{len(failed_models)}</p>
                <p class="metric-label">Modèles échoués</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <p class="metric-value">0</p>
                <p class="metric-label">Modèles échoués</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Modèles disponibles
    st.subheader("Modèles Disponibles")
    
    if loaded_models:
        cols = st.columns(3)
        
        for i, model_name in enumerate(loaded_models):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card" style="padding: 1rem; border-left: 3px solid {COLORS['accent_medium']};">
                    <p style="font-weight: 600; color: {COLORS['text_dark']}; margin: 0; font-size: 1rem;">{model_name}</p>
                    <p style="color: {COLORS['text_light']}; margin-top: 0.25rem; font-size: 0.85rem;">Prêt à l'emploi</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">Aucun modèle chargé</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Features analysées
    st.subheader("Features Analysées")
    
    features_info = {
        "speed": "Vitesse du véhicule (km/h)",
        "relative_speed": "Vitesse relative par rapport aux autres véhicules (km/h)",
        "avg_traffic_speed": "Vitesse moyenne du trafic (km/h)",
        "acceleration": "Accélération instantanée (m/s²)",
        "distance_to_nearest": "Distance au véhicule le plus proche (m)",
        "direction_change": "Changement de direction angulaire (degrés)"
    }
    
    for feature, description in features_info.items():
        st.markdown(f"**{feature}**")
        st.markdown(f"<p style='color: {COLORS['text_light']}; margin-top: -0.5rem; margin-bottom: 1rem;'>{description}</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Classes de prédiction avec cartes colorées douces
    st.subheader("Classes de Prédiction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="class-card-normal">
            <h4 style="color: {COLORS['normal_dark']}; margin: 0 0 0.5rem 0; font-weight: 600;">Normal</h4>
            <p style="color: {COLORS['normal_dark']}; margin: 0; font-size: 0.9rem; opacity: 0.9;">
                Conduite sûre et conforme aux règles de circulation
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="class-card-suspect">
            <h4 style="color: {COLORS['suspect_dark']}; margin: 0 0 0.5rem 0; font-weight: 600;">Suspect</h4>
            <p style="color: {COLORS['suspect_dark']}; margin: 0; font-size: 0.9rem; opacity: 0.9;">
                Comportement inhabituel nécessitant surveillance
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="class-card-danger">
            <h4 style="color: {COLORS['danger_dark']}; margin: 0 0 0.5rem 0; font-weight: 600;">Dangereux</h4>
            <p style="color: {COLORS['danger_dark']}; margin: 0; font-size: 0.9rem; opacity: 0.9;">
                Comportement à risque nécessitant intervention
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Chemins des fichiers
    st.subheader("Chemins des Fichiers")
    
    file_info = f"""
    <div class="info-box">
        <strong style="color: {COLORS['text_dark']};">Structure du projet :</strong><br>
        • <strong>Dataset :</strong> {config.CSV_FEATURES}<br>
        • <strong>Vidéo originale :</strong> {config.VIDEO_INPUT}<br>
        • <strong>Vidéo annotée :</strong> {config.VIDEO_OUTPUT}<br>
        • <strong>Dossier modèles :</strong> {config.MODELS_DIR}
    </div>
    """
    st.markdown(file_info, unsafe_allow_html=True)

