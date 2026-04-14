import streamlit as st
import numpy as np
import pickle
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Prediksi Kualitas Udara Jakarta",
    page_icon="🌫️",
    layout="wide"
)

# ========================================== 
# 2. LOAD SEMUA ASSET & MODEL
# ========================================== 
@st.cache_resource
def load_all_assets():
    assets = {}
    try:
        assets['ann'] = load_model('ann_model.h5')
        
        with open('scaler.pkl', 'rb') as f:
            assets['scaler'] = pickle.load(f)
            
        with open('label_encoder.pkl', 'rb') as f:
            assets['encoder'] = pickle.load(f)
            
        with open('fis_manual_config.json', 'r') as f:
            assets['fis_manual'] = json.load(f)
            
        with open('fis_ga_config.json', 'r') as f:
            assets['fis_ga'] = json.load(f)
            
    except Exception as e:
        st.error(f"⚠️ Gagal memuat file model: {e}. Pastikan semua file ada di folder yang sama!")
        
    return assets

assets = load_all_assets()

# ==========================================
# 3. LOGIKA FIS MAMDANI PURE NUMPY
# ==========================================
U_PM25 = np.arange(0, 301, 1)   
U_PM10 = np.arange(0, 201, 1)   
U_CO   = np.arange(0, 51, 0.5)  
U_OUT  = np.arange(0, 101, 1)   

def trimf(x, abc):
    a, b, c = abc
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def fuzzify(val, universe, mf_params):
    result = {}
    for name, params in mf_params.items():
        result[name] = float(trimf(np.array([val]), params)[0])
    return result

def centroid(agg_output, universe):
    if np.sum(agg_output) == 0:
        return np.mean(universe)
    return np.sum(universe * agg_output) / np.sum(agg_output)

def _run_fis(pm25_val, pm10_val, co_val, config_dict):
    """Menjalankan FIS berdasarkan dictionary config (manual/GA)"""
    mf_pm25 = config_dict['mf_pm25']
    mf_pm10 = config_dict['mf_pm10']
    mf_co   = config_dict['mf_co']
    mf_out  = config_dict['mf_out']
    rules   = config_dict['rules']

    # 1. Fuzzifikasi
    f25 = fuzzify(pm25_val, U_PM25, mf_pm25)
    f10 = fuzzify(pm10_val, U_PM10, mf_pm10)
    fco = fuzzify(co_val,   U_CO,   mf_co)

    # 2. Evaluasi Rule & Agregasi
    agg = np.zeros(len(U_OUT))
    for rule in rules:
        p25_t, p10_t, co_t, out_t = rule
        # Ambil nilai minimum (AND operator)
        strength = min(f25.get(p25_t, 0), f10.get(p10_t, 0), fco.get(co_t, 0))
        
        if strength > 0:
            # Potong output MF (Mamdani Implication)
            out_mf = np.minimum(strength, trimf(U_OUT, mf_out[out_t]))
            # Gabungkan (Agregasi MAX)
            agg = np.maximum(agg, out_mf)

    # 3. Defuzzifikasi
    score = centroid(agg, U_OUT)
    return score

def score_to_label(score):
    if score <= 25: return 'Sangat Aman'
    elif score <= 45: return 'Aman'
    elif score <= 65: return 'Netral' 
    elif score <= 85: return 'Tidak Sehat'
    else: return 'Berbahaya'

# ==========================================
# 4. PREDIKSI ANN
# ==========================================
def predict_ann(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    
    # Normalisasi Data
    input_scaled = assets['scaler'].transform(input_array)
        
    # Prediksi
    prediction = assets['ann'].predict(input_scaled)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Decode Label
    label = assets['encoder']['classes'][class_idx]
    return label, confidence

# ==========================================
# 5. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("🌫️ Sistem Penilaian Risiko Kualitas Udara Jakarta")
st.markdown("**Proyek UTS Soft Computing: The Intelligence Battle (Human vs GA vs ANN)**")
st.markdown("---")

tab1, tab2 = st.tabs(["🚀 Prediksi Kualitas Udara", "📊 Analisis & Ablation Study (GA)"])

# ---------------- TAB 1: PREDIKSI ----------------
with tab1:
    st.sidebar.header("Input Parameter Polutan")
    st.sidebar.write("Masukkan nilai sensor udara:")

    pm25 = st.sidebar.slider("PM2.5 (µg/m³)", 0.0, 300.0, 50.0)
    pm10 = st.sidebar.slider("PM10 (µg/m³)", 0.0, 200.0, 50.0)
    co   = st.sidebar.slider("CO (ppm)", 0.0, 50.0, 15.0)
    
    st.sidebar.markdown("---")
    st.sidebar.write("*Hanya digunakan oleh model ANN:*")
    so2  = st.sidebar.slider("SO2", 0.0, 150.0, 20.0)
    o3   = st.sidebar.slider("O3", 0.0, 350.0, 30.0)
    no2  = st.sidebar.slider("NO2", 0.0, 250.0, 10.0)

    st.subheader("Pilih Kecerdasan Pengambil Keputusan")
    model_choice = st.radio(
        "Metode apa yang ingin digunakan?",
        ("Pakar Manusia (Manual FIS)", "Optimasi Evolusioner (FIS + GA)", "Murni Mesin (ANN)"),
        horizontal=True
    )

    if st.button("Analisis Kualitas Udara", type="primary", use_container_width=True):
        st.markdown("### Hasil Analisis")
        
        if "ANN" in model_choice:
            input_data = [pm25, pm10, so2, co, o3, no2]
            label, conf = predict_ann(input_data)
            
            col1, col2 = st.columns(2)
            col1.metric(label="Kategori Kualitas Udara (ANN)", value=label)
            col2.metric(label="Confidence", value=f"{conf:.2f}%")
            
        elif "GA" in model_choice:
            score = _run_fis(pm25, pm10, co, assets['fis_ga'])
            label = score_to_label(score)
            
            col1, col2 = st.columns(2)
            col1.metric(label="Kategori Kualitas Udara (FIS + GA)", value=label)
            col2.metric(label="Skor FIS Crisp", value=f"{score:.2f}")
            
        else:
            score = _run_fis(pm25, pm10, co, assets['fis_manual'])
            label = score_to_label(score)
            
            col1, col2 = st.columns(2)
            col1.metric(label="Kategori Kualitas Udara (Pakar Manusia)", value=label)
            col2.metric(label="Skor FIS Crisp", value=f"{score:.2f}")

        # Peringatan Visual berdasarkan Output Label
        if label in ['Sangat Aman', 'Aman', 'Netral']:
            st.success("Udara dalam batas yang dapat ditoleransi.")
        else:
            st.error("Peringatan! Udara berisiko bagi kesehatan.")

# ---------------- TAB 2: ABLATION STUDY ----------------
with tab2:
    st.header("Hasil Eksperimen (Ablation Study GA)")
    st.write("Visualisasi pergeseran kurva dan konvergensi Algoritma Genetika saat mencari parameter optimal.")
    try:
        # Menampilkan gambar yang diupload
        st.image("ga_convergence.png", use_container_width=True)
    except:
        st.info("⚠️ Gambar 'ga_convergence.png' belum ditemukan. Pastikan kamu sudah me-rename gambar dari Colab menjadi 'ga_convergence.png' di folder ini.")