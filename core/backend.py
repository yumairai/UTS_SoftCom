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
    models = {'ann': None, 'scaler': None, 'label_enc': {'classes': CLASS_ORDER}, 'fis_manual': None, 'fis_ga': None}
    
    try:
        # PERBAIKAN 1: Load kedua konfigurasi JSON
        if os.path.exists('fis_manual_config.json'):
            models['fis_manual'] = json.load(open('fis_manual_config.json'))
        if os.path.exists('fis_ga_config.json'):
            models['fis_ga'] = json.load(open('fis_ga_config.json'))
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

# --- PERBAIKAN 2: Logika FIS Real (Mamdani) ---
def predict_fis(pm25, pm10, co, config=None):
    # 1. Ekstrak Parameter dari Config (Atau gunakan default jika config kosong)
    if config:
        mf_pm25 = config.get('mf_pm25', MF_PM25_DEFAULT)
        mf_pm10 = config.get('mf_pm10', MF_PM10_DEFAULT)
        mf_co   = config.get('mf_co', MF_CO_DEFAULT)
        mf_out  = config.get('mf_out', MF_OUT_DEFAULT)
        rules   = config.get('rules', RULES_DEFAULT)
    else:
        mf_pm25, mf_pm10, mf_co, mf_out, rules = MF_PM25_DEFAULT, MF_PM10_DEFAULT, MF_CO_DEFAULT, MF_OUT_DEFAULT, RULES_DEFAULT

    # 2. Tahap Fuzzifikasi (Menghitung derajat keanggotaan input di setiap himpunan fuzzy)
    fuzz_pm25 = fuzzify(pm25, mf_pm25)
    fuzz_pm10 = fuzzify(pm10, mf_pm10)
    fuzz_co   = fuzzify(co, mf_co)

    # Simpan status aktif ini agar bisa di-passing ke UI jika perlu
    mf_active = {'pm25': fuzz_pm25, 'pm10': fuzz_pm10, 'co': fuzz_co}

    # 3. Tahap Inferensi & Evaluasi Aturan (Mamdani MIN-MAX)
    agg_output = np.zeros_like(U_OUT, dtype=float)

    for rule in rules:
        if len(rule) == 4:
            cat25, cat10, cat_co, cat_out = rule
            
            # Cari nilai minimal (operator AND) dari himpunan yang terpanggil di aturan ini
            activation = min(
                fuzz_pm25.get(cat25, 0.0),
                fuzz_pm10.get(cat10, 0.0),
                fuzz_co.get(cat_co, 0.0)
            )

            # Jika rule ini aktif (nilai di atas 0)
            if activation > 0:
                out_params = mf_out.get(cat_out)
                if out_params:
                    # Gambar kurva output untuk kategori tersebut (Sangat Aman, Aman, dll)
                    out_mf_array = np.array([trimf(x, tuple(out_params)) for x in U_OUT])
                    
                    # Potong kurva berdasarkan seberapa kuat aktivasinya (Implikasi MIN)
                    implied_mf = np.minimum(activation, out_mf_array)
                    
                    # Gabungkan dengan kurva agregasi utama (Agregasi MAX)
                    agg_output = np.maximum(agg_output, implied_mf)

    # 4. Tahap Defuzzifikasi (Menghitung nilai tegas dari gabungan area kurva)
    score = centroid_defuzz(agg_output, U_OUT)

    # 5. Penentuan Label Kategorikal (Klasifikasi)
    # Cek di kurva mana (Sangat Aman, Aman, Tidak Sehat, Berbahaya) skor ini berada paling tinggi
    best_label = "Unknown"
    max_membership = -1.0
    for cls_name, params in mf_out.items():
        deg = trimf(score, tuple(params))
        if deg > max_membership:
            max_membership = deg
            best_label = cls_name

    # Konversi label "Netral" (jika di JSON ada) menjadi "Tidak Sehat" agar sesuai dengan 4 kelas ANN
    if best_label == "Netral":
        best_label = "Tidak Sehat"

    return best_label, score, mf_active, mf_pm25, mf_pm10, mf_co

def predict_ann(pm25, pm10, co, models):
    classes = models['label_enc']['classes']
    if models['ann'] is not None and models['scaler'] is not None:
        X = np.array([[pm25, pm10, co]])
        X_scaled = models['scaler'].transform(X)
        proba = models['ann'].predict(X_scaled, verbose=0)[0]
        label = classes[int(np.argmax(proba))]
        return label, proba
    return "Unknown", np.array([0.25, 0.25, 0.25, 0.25])