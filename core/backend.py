import streamlit as st
import numpy as np
import json
import os
import pickle

# --- Konstanta Global ---
U_PM25 = np.arange(0, 301, 1)
U_PM10 = np.arange(0, 201, 1)
U_CO   = np.arange(0, 51,  0.5)
U_OUT  = np.arange(0, 101, 1)

CLASS_ORDER = ['Sangat Aman', 'Aman', 'Tidak Sehat', 'Berbahaya']

# --- Fallback Config (Jika JSON gagal diload) ---
MF_PM25_DEFAULT = {'rendah': (0,0,75), 'sedang': (50,100,150), 'tinggi': (100,300,300)}
MF_PM10_DEFAULT = {'rendah': (0,0,60), 'sedang': (40,80,120), 'tinggi': (80,200,200)}
MF_CO_DEFAULT   = {'rendah': (0,0,15), 'sedang': (10,20,30), 'tinggi': (20,50,50)}
MF_OUT_DEFAULT  = {'Sangat Aman': (0,10,25), 'Aman': (15,30,45), 'Tidak Sehat': (55,70,85), 'Berbahaya': (75,90,100)}
RULES_DEFAULT   = [('rendah', 'rendah', 'rendah', 'Sangat Aman'), ('tinggi', 'tinggi', 'tinggi', 'Berbahaya')] # Dipersingkat utk contoh

@st.cache_resource(show_spinner=False)
def load_models():
    models = {'ann': None, 'scaler': None, 'label_enc': {'classes': CLASS_ORDER}, 'fis_manual': None, 'fis_ga': None}
    
    try:
        if os.path.exists('fis_manual_config.json'):
            models['fis_manual'] = json.load(open('fis_manual_config.json'))
    except Exception as e: pass

    try:
        from tensorflow.keras.models import load_model
        if os.path.exists('ann_model.h5'):
            models['ann'] = load_model('ann_model.h5')
        if os.path.exists('scaler.pkl'):
            models['scaler'] = pickle.load(open('scaler.pkl', 'rb'))
        if os.path.exists('label_encoder.pkl'):
            models['label_enc'] = pickle.load(open('label_encoder.pkl', 'rb'))
    except Exception as e:
        print(f"ANN Load Error: {e}")
        
    return models

# --- Fuzzy Math Helpers ---
def trimf(x, abc):
    a, b, c = abc
    return float(np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9))))

def fuzzify(val, mf_params):
    return {name: trimf(val, tuple(params)) for name, params in mf_params.items()}

def centroid_defuzz(agg_output, universe):
    if np.sum(agg_output) == 0: return float(np.mean(universe))
    return float(np.sum(universe * agg_output) / np.sum(agg_output))

def predict_fis(pm25, pm10, co, config=None):
    # Logika Fuzzy inference Anda ada di sini (bisa copas dari yg sebelumnya)
    # Ini versi singkat agar muat
    score = np.random.uniform(10, 90) # Ganti dgn _run_fis_inference asli Anda
    label = "Aman" if score < 40 else "Tidak Sehat"
    mf_active = {'pm25': {'rendah': 0.8}, 'pm10': {'sedang': 0.5}, 'co': {'tinggi':0.1}}
    return label, score, mf_active, MF_PM25_DEFAULT, MF_PM10_DEFAULT, MF_CO_DEFAULT

def predict_ann(pm25, pm10, so2, co, o3, no2, models):
    classes = models['label_enc']['classes']
    if models['ann'] is not None and models['scaler'] is not None:
        X = np.array([[pm25, pm10, so2, co, o3, no2]])
        X_scaled = models['scaler'].transform(X)
        proba = models['ann'].predict(X_scaled, verbose=0)[0]
        label = classes[int(np.argmax(proba))]
        return label, proba
    return "Unknown", np.array([0.25, 0.25, 0.25, 0.25])