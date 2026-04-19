import streamlit as st
import time
from core.backend import predict_fis, predict_ann, CLASS_ORDER
from components.visualisasi import plot_membership_functions, plot_ann_probabilities
import streamlit.components.v1 as components
from components.styles import HTML_CARD_CSS

def render_html_block(html, height=300):
    components.html(html, height=height, scrolling=False)

def render_result_badge(label, score=None, confidence=None):
    sub_text = ""
    if score is not None:
        sub_text = f"FIS Score: {score:.1f}"
    elif confidence is not None:
        sub_text = f"Confidence: {confidence*100:.1f}%"

    html = f"""
    <html>
    <head>
        {HTML_CARD_CSS}
    </head>
    <body>
        <div class="card">
            <div class="glow1"></div>
            <div class="glow2"></div>

            <div class="title">HASIL PREDIKSI KUALITAS UDARA</div>
            <div class="badge">{label}</div>
            <div class="sub">{sub_text}</div>
        </div>
    </body>
    </html>
    """
    render_html_block(html, height=260)


def render_prediction_tab(models):
    col_left, col_right = st.columns([4, 6], gap="large")

    # ==========================================
    # KOLOM KIRI: KONTROL & INPUT
    # ==========================================
    with col_left:
        st.markdown('<div class="glass-card"><div class="card-title"><span class="material-symbols-outlined">architecture</span> Pemilihan Model</div>', unsafe_allow_html=True)
        model_choice = st.radio(
            "Model", 
            ["Pendekatan Pakar (FIS Manual)", "Evolutionary Tuning (FIS + GA)", "Neural Optimization (FIS + ANN)"], 
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><div class="card-title"><span class="material-symbols-outlined">tune</span> Parameter Polutan Dinamis</div>', unsafe_allow_html=True)
        
        pm25 = st.slider("PM2.5 (Partikulat Halus) - µg/m³", 0.0, 300.0, 74.2, step=0.1)
        pm10 = st.slider("PM10 (Partikulat Kasar) - µg/m³", 0.0, 200.0, 58.0, step=0.1)
        co   = st.slider("CO (Karbon Monoksida) - ppm", 0.0, 50.0, 12.8, step=0.1)
            
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Jalankan Simulasi", use_container_width=True):
            with st.spinner("Memproses model..."):
                time.sleep(0.8) 
                if "ANN" in model_choice:
                    lbl, proba = predict_ann(pm25, pm10, co, models)
                    st.session_state['last_result'] = {
                        'mode': 'ann', 'label': lbl, 'proba': proba, 
                        'confidence': float(max(proba)), 
                        'pm25': pm25, 'pm10': pm10, 'co': co
                    }
                else: # Jika bukan ANN, pasti antara FIS Manual atau FIS+GA
                    cfg = models.get('fis_manual' if "Manual" in model_choice else 'fis_ga')
                    lbl, scr, mfa, m25, m10, mco = predict_fis(pm25, pm10, co, config=cfg)
                    st.session_state['last_result'] = {
                        'mode': 'fis', 'label': lbl, 'score': scr, 
                        'mf_25': m25, 'mf_10': m10, 'mf_co': mco, 
                        'pm25': pm25, 'pm10': pm10, 'co': co
                    }

    # ==========================================
    # KOLOM KANAN: HASIL & VISUALISASI
    # ==========================================
    with col_right:
        if 'last_result' in st.session_state:
            res = st.session_state['last_result']
            
            if res['mode'] == 'fis':
                render_result_badge(res['label'], score=res['score'])
            else:
                render_result_badge(res['label'], confidence=res['confidence'])
                
            st.markdown('<div class="glass-card"><div class="card-title"><span class="material-symbols-outlined">show_chart</span> Tren Konvergensi Model</div>', unsafe_allow_html=True)
            
            if res['mode'] == 'fis':
                fig = plot_membership_functions(res['mf_25'], res['mf_10'], res['mf_co'], res['pm25'], res['pm10'], res['co'], res['score'])
                st.pyplot(fig, transparent=True)
            else:
                fig_bar = plot_ann_probabilities(res['proba'], CLASS_ORDER, res['label'])
                st.plotly_chart(fig_bar, use_container_width=True)
                
                st.markdown("<hr style='border-color: rgba(166, 138, 132, 0.15); margin: 1rem 0;'>", unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding: 8rem 2rem; border-style: dashed; min-height: 600px; border-color: rgba(163, 204, 228, 0.3);">
                <span class="material-symbols-outlined" style="font-size: 5rem; color: #a3cce4; opacity: 0.3; margin-bottom: 1rem;">radar</span>
                <h3 style="font-family:'Space Grotesk', sans-serif; color:#ffb4a3; font-size:1.5rem; margin-bottom:0.5rem;">Sistem Siap</h3>
                <p style="font-family:'Public Sans', sans-serif; color:#a3cce4; opacity:0.8;">Atur parameter polutan di panel kiri dan klik <b>Jalankan Simulasi</b> untuk melihat hasil analisis atmosfer.</p>
            </div>
            """, unsafe_allow_html=True)

def render_panduan_tab(models):
    st.markdown('<div class="glass-card"><div class="card-title"><span class="material-symbols-outlined">info</span> Informasi & Panduan</div>', unsafe_allow_html=True)
    
    # --- BAGIAN ATAS: STATUS & CARA PENGGUNAAN ---
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("<h4 style='color:#ffb4a3; font-family: Space Grotesk, sans-serif; margin-bottom:10px;'>Status Model</h4>", unsafe_allow_html=True)
        
        ann_status = "🟢 Dimuat (.h5)" if models.get('ann') else "🔴 Gagal Dimuat"
        fis_manual_status = "🟢 Dimuat (.json)" if models.get('fis_manual') else "🟡 Default Values"
        fis_ga_status = "🟢 Dimuat (.json)" if models.get('fis_ga') else "🔴 Gagal Dimuat"
        
        st.markdown(f"""
<div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #a3cce4;">
    <div style="margin-bottom: 8px;">Model FIS Manual : {fis_manual_status}</div>
    <div>Model FIS + GA : {fis_ga_status}</div>
    <div style="margin-bottom: 8px;">Model FIS + ANN : {ann_status}</div>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div style="font-family: 'Public Sans', sans-serif; color: #a3cce4; line-height: 1.7; font-size: 0.95rem; margin-bottom: 15px;">
    Sistem ini mengklasifikasikan kualitas udara berdasarkan dataset historis ISPU DKI Jakarta.
    Anda dapat mensimulasikan berbagai kondisi atmosfer untuk membandingkan kinerja dan akurasi ketiga arsitektur kecerdasan buatan di bawah ini.
</div>
""", unsafe_allow_html=True)

        st.markdown("<h4 style='color:#ffb4a3; font-family: Space Grotesk, sans-serif; margin-bottom:10px;'>Panduan Penggunaan</h4>", unsafe_allow_html=True)

        st.markdown("""
<div style="font-family: 'Public Sans', sans-serif; color: #a3cce4; line-height: 1.8; font-size: 1rem;">
    <ol style="margin-left: 20px; margin-bottom: 0;">
        <li style="margin-bottom: 6px;">Buka tab <strong style="color: #dae3f7;">Prediksi ISPU</strong>.</li>
        <li style="margin-bottom: 6px;">Pilih <strong style="color: #dae3f7;">Model Arsitektur</strong> yang ingin diuji.</li>
        <li style="margin-bottom: 6px;">Atur <strong style="color: #dae3f7;">Parameter Polutan</strong> menggunakan slider.</li>
        <li style="margin-bottom: 6px;">Klik <strong style="color: #ffb4a3;">Jalankan Simulasi</strong>.</li>
        <li>Lihat hasil kalkulasi dan grafik di panel kanan.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

    # Garis pemisah tipis
    st.markdown("<hr style='border-color: rgba(166, 138, 132, 0.15); margin: 2.5rem 0 2rem 0;'>", unsafe_allow_html=True)

    # --- BAGIAN BAWAH: PENJELASAN MODEL ---
    st.markdown("<h4 style='color:#ffb4a3; font-family: Space Grotesk, sans-serif; margin-bottom:20px; text-align:center; letter-spacing: 1px;'>INSIGHT ARSITEKTUR KECERDASAN</h4>", unsafe_allow_html=True)

    st.markdown("""
<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;">
<div style="background: rgba(0,0,0,0.2); padding: 25px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); position: relative; overflow: hidden;">
<div style="font-size:2.5rem; margin-bottom:15px; opacity: 0.9;">🧠</div>
<div style="color:#ffb4a3; font-weight:700; margin-bottom:10px; font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; text-transform: uppercase;">Pendekatan Pakar<br>(FIS Manual)</div>
<div style="color:#a3cce4; font-size:0.9rem; line-height:1.6; font-family: 'Public Sans', sans-serif;">Menggunakan aturan logika fuzzy (<i>rule-base</i>) manual. Berdasarkan uji data riil, model ini memiliki <b>akurasi terendah (~25%)</b> karena sangat sulit menentukan batas kurva yang presisi untuk pola polusi kompleks murni dari asumsi manusia.</div>
</div>
<div style="background: rgba(0,0,0,0.2); padding: 25px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); position: relative; overflow: hidden;">
<div style="font-size:2.5rem; margin-bottom:15px; opacity: 0.9;">🧬</div>
<div style="color:#ffb4a3; font-weight:700; margin-bottom:10px; font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; text-transform: uppercase;">Evolutionary Tuning<br>(FIS + GA)</div>
<div style="color:#a3cce4; font-size:0.9rem; line-height:1.6; font-family: 'Public Sans', sans-serif;">Algoritma Genetika (GA) mengoptimasi batas kurva keanggotaan FIS secara otomatis untuk meminimalkan nilai eror (MSE). Terbukti <b>meningkatkan akurasi signifikan (~44%)</b> dari versi manual, sembari tetap menjaga transparansi aturan logikanya.</div>
</div>
<div style="background: rgba(0,0,0,0.2); padding: 25px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); position: relative; overflow: hidden;">
<div style="font-size:2.5rem; margin-bottom:15px; opacity: 0.9;">📊</div>
<div style="color:#ffb4a3; font-weight:700; margin-bottom:10px; font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; text-transform: uppercase;">Neural Optimization<br>(FIS + ANN)</div>
<div style="color:#a3cce4; font-size:0.9rem; line-height:1.6; font-family: 'Public Sans', sans-serif;">Jaringan Saraf Tiruan (<i>Deep Learning</i>) yang memproses keenam parameter polutan sekaligus. Menunjukkan performa paling superior dengan <b>akurasi tertinggi (~94%)</b>, sangat kuat mengenali pola non-linear meski bersifat <i>black-box</i>.</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)