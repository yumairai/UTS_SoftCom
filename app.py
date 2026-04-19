import streamlit as st
from components.styles import inject_css
from core.backend import load_models

# Tambahkan import render_panduan_tab
from components.ui_tabs import render_prediction_tab, render_panduan_tab

# Konfigurasi halaman dasar - Set sidebar ke collapsed sejak awal
st.set_page_config(
    page_title="Sistem Klasifikasi Kualitas Udara (ISPU)", 
    page_icon="🌫️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    # 1. CSS Custom
    inject_css()

    # 2. Muat model backend 
    with st.spinner("Memuat model AI..."):
        models = load_models()

    # 3. Header Utama 
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title"> Sistem Pendeteksi Kualitas Udara</h1>
        <p class="main-subtitle">
            Berbasis Indeks Standar Pencemar Udara (ISPU) DKI Jakarta <br>
            Perbandingan: Pakar Manusia vs. Evolutionary Tuning & Neuro-Fuzzy
        </p>
    </div>
    <br>
    """, unsafe_allow_html=True)

    # 4. Routing Halaman Utama 
    tab1, tab2 = st.tabs([
        "🔬 Prediksi ISPU", 
        "ℹ️ Status & Panduan"
    ])
    
    with tab1:
        render_prediction_tab(models)
    with tab2:
        render_panduan_tab(models)

if __name__ == "__main__":
    main()