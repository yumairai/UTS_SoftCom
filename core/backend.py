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
RULES_DEFAULT   = [('rendah', 'rendah', 'rendah', 'Sangat Aman'), ('tinggi', 'tinggi', 'tinggi', 'Berbahaya')] 

@st.cache_resource(show_spinner=False)
def load_models():
    models = {
        'ann': None, 
        'scaler': None, 
        'label_enc': {'classes': CLASS_ORDER}, 
        'fis_manual': None, 
        'fis_ga': None
    }
    
    # 1. Cek & Load Config JSON
    try:
        if os.path.exists('fis_manual_config.json'):
            with open('fis_manual_config.json', 'r') as f:
                models['fis_manual'] = json.load(f)
        
        if os.path.exists('fis_ga_config.json'):
            with open('fis_ga_config.json', 'r') as f:
                models['fis_ga'] = json.load(f)
    except Exception as e:
        st.error(f"Gagal memuat JSON: {e}")

    # 2. Cek & Load Model ANN + Scaler
    try:
        from tensorflow.keras.models import load_model
        
        # Cek ANN
        if os.path.exists('ann_model.h5'):
            models['ann'] = load_model('ann_model.h5')
        else:
            st.error("File 'ann_model.h5' tidak ditemukan!")

        # Cek Scaler
        if os.path.exists('scaler.pkl'):
            with open('scaler.pkl', 'rb') as f:
                models['scaler'] = pickle.load(f)
        else:
            st.error("File 'scaler.pkl' tidak ditemukan!")

        # Cek Label Encoder
        if os.path.exists('label_encoder.pkl'):
            with open('label_encoder.pkl', 'rb') as f:
                models['label_enc'] = pickle.load(f)
                
    except Exception as e:
        # Jika muncul error "Unknown layer", "Bad magic number", dsb
        st.error(f"⚠️ Error Kritikal saat memuat model: {e}")
        st.info("Saran: Pastikan versi TensorFlow di laptop sama dengan di Colab.")
        
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

# --- PERBAIKAN LOGIKA FIS (Agar Tidak Unknown) ---
def predict_fis(pm25, pm10, co, config=None):
    # 1. Ekstrak Parameter
    if config:
        mf_pm25 = config.get('mf_pm25', MF_PM25_DEFAULT)
        mf_pm10 = config.get('mf_pm10', MF_PM10_DEFAULT)
        mf_co   = config.get('mf_co', MF_CO_DEFAULT)
        mf_out  = config.get('mf_out', MF_OUT_DEFAULT)
        rules   = config.get('rules', RULES_DEFAULT)
    else:
        mf_pm25, mf_pm10, mf_co, mf_out, rules = MF_PM25_DEFAULT, MF_PM10_DEFAULT, MF_CO_DEFAULT, MF_OUT_DEFAULT, RULES_DEFAULT

    # 2. Fuzzifikasi
    fuzz_pm25 = fuzzify(pm25, mf_pm25)
    fuzz_pm10 = fuzzify(pm10, mf_pm10)
    fuzz_co   = fuzzify(co, mf_co)
    mf_active = {'pm25': fuzz_pm25, 'pm10': fuzz_pm10, 'co': fuzz_co}

    # 3. Inferensi
    agg_output = np.zeros_like(U_OUT, dtype=float)
    rule_fired = False

    for rule in rules:
        if len(rule) == 4:
            cat25, cat10, cat_co, cat_out = rule
            activation = min(
                fuzz_pm25.get(cat25, 0.0),
                fuzz_pm10.get(cat10, 0.0),
                fuzz_co.get(cat_co, 0.0)
            )

            if activation > 0:
                rule_fired = True
                out_params = mf_out.get(cat_out)
                if out_params:
                    out_mf_array = np.array([trimf(x, tuple(out_params)) for x in U_OUT])
                    implied_mf = np.minimum(activation, out_mf_array)
                    agg_output = np.maximum(agg_output, implied_mf)

    # 4. Defuzzifikasi & Fallback
    # Jika tidak ada rule yang kena, kita hitung skor manual berdasarkan rata-rata input
    if not rule_fired:
        # Fallback: Skor dihitung dari rata-rata persentase input terhadap ambang batas
        score = (min(pm25/300, 1) + min(pm10/200, 1) + min(co/50, 1)) / 3 * 100
    else:
        score = centroid_defuzz(agg_output, U_OUT)

    # 5. Penentuan Label (Klasifikasi)
    best_label = "Sangat Aman" # Default fallback terendah
    max_membership = -1.0
    
    for cls_name, params in mf_out.items():
        deg = trimf(score, tuple(params))
        if deg > max_membership:
            max_membership = deg
            best_label = cls_name
    
    # Jika skor sangat tinggi tapi tidak kena MF manapun (out of range)
    if score > 80 and max_membership <= 0:
        best_label = "Berbahaya"

    # Konsistensi Label
    if best_label == "Netral": best_label = "Tidak Sehat"
    
    return best_label, score, mf_active, mf_pm25, mf_pm10, mf_co

# --- PERBAIKAN LOGIKA ANN ---
def predict_ann(pm25, pm10, co, models):
    # Pastikan classes diambil dari CLASS_ORDER agar tidak Unknown
    classes = CLASS_ORDER 
    
    if models['ann'] is not None and models['scaler'] is not None:
        try:
            X = np.array([[pm25, pm10, co]])
            X_scaled = models['scaler'].transform(X)
            proba = models['ann'].predict(X_scaled, verbose=0)[0]
            label = classes[int(np.argmax(proba))]
            return label, proba
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            
    # Jika model gagal load, berikan prediksi dummy berdasarkan input sederhana
    # agar UI tidak pecah/Unknown
    dummy_idx = 0
    if pm25 > 150 or pm10 > 100: dummy_idx = 2
    if pm25 > 250: dummy_idx = 3
    
    return classes[dummy_idx], np.array([0.0, 0.0, 0.0, 0.0])