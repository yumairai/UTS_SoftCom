import streamlit as st

# Konstanta CSS untuk HTML Card (Badge & Kesimpulan) agar tidak mengotori ui_tabs.py
HTML_CARD_CSS = """
<style>
    body {
        margin: 0;
        background: transparent;
        font-family: 'Public Sans', sans-serif;
    }

    .card {
        position: relative;
        overflow: hidden;
        text-align: center;
        padding: 3rem 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 180, 163, 0.3);
        background: rgba(20, 30, 50, 0.6);
        backdrop-filter: blur(20px);
    }

    .glow1 {
        position: absolute;
        top: -100px;
        right: -100px;
        width: 250px;
        height: 250px;
        background: rgba(255, 180, 163, 0.2);
        filter: blur(60px);
        border-radius: 50%;
    }

    .glow2 {
        position: absolute;
        bottom: -100px;
        left: -100px;
        width: 250px;
        height: 250px;
        background: rgba(163, 204, 228, 0.2);
        filter: blur(60px);
        border-radius: 50%;
    }

    .title {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.3em;
        color: #a3cce4;
        margin-bottom: 1rem;
    }

    .badge {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        color: #ffb4a3;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .sub {
        font-size: 1rem;
        color: #dae3f7;
    }
    
    /* CSS Khusus Bento Grid Kesimpulan */
    .bento-container {
        display: grid;
        grid-template-columns: 1.3fr 1fr;
        gap: 1.5rem;
    }

    .bento-left { display: flex; }
    .bento-right {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }

    .bento-card {
        position: relative;
        padding: 24px;
        border-radius: 16px;
        background: rgba(20, 30, 50, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        height: 100%;
        overflow: hidden;
    }

    .bento-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffb4a3;
        margin-bottom: 10px;
    }

    .bento-text {
        font-size: 0.95rem;
        color: #a3cce4;
        line-height: 1.6;
    }

    .bento-icon {
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 2.5rem;
        opacity: 0.15;
    }

    .bento-highlight {
        border: 1px solid rgba(255,180,163,0.3);
    }
</style>
"""

def inject_css():
    st.markdown("""
    <style>
    /* Import Font dari Stitch */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700;800&family=Public+Sans:wght@300;400;500;700&family=Inter:wght@400;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

    :root {
        --bg-main: #0a1421;
        --bg-glass: rgba(44, 53, 68, 0.4);
        --border-glass: rgba(166, 138, 132, 0.15);
        --primary: #ffb4a3;
        --primary-dark: #ee6c4d;
        --secondary: #a3cce4;
        --text-main: #dae3f7;
        --text-muted: #92bbd3;
        --surface-low: #131c2a;
    }

    /* ── Global Background ── */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Public Sans', sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
    }
    
    #MainMenu, footer, header, [data-testid="collapsedControl"], [data-testid="stSidebar"] { display: none !important; }
    
    /* ── Header ── */
    .main-header {
        padding: 10px 0 20px;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        color: #F3F4F6;
        letter-spacing: -0.5px;
        font-family: 'Space Grotesk', sans-serif;
        text-align: center;
    }

    .main-subtitle {
        margin-top: 8px;
        color: #00D2FF;
        font-size: 1.1rem;
        font-weight: 500;
        text-align: center;
    }
    
    /* Paksa warna teks Streamlit */
    .stMarkdown p, .stMarkdown li, div[role="radiogroup"] label p {
        color: var(--text-main) !important;
        font-family: 'Public Sans', sans-serif !important;
    }

    /* ── Glassmorphism Cards (Stitch Style) ── */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--secondary);
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Tombol Gradient (Run Prediction) ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: #3d0700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 20px !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 20px rgba(238, 108, 77, 0.3) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 30px rgba(238, 108, 77, 0.5) !important;
    }

    /* Memposisikan Spinner ke Tengah */
    div[data-testid="stSpinner"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
    }
    
    div[data-testid="stSpinner"] > div {
        color: var(--primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
    }

    /* ── Sliders (Meniru input range Stitch) ── */
    div[data-testid="stSlider"] > div > div > div > div {
        background-color: var(--primary) !important;
    }
    div[data-testid="stSlider"] label p {
        font-family: 'Inter', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-muted) !important;
        font-size: 0.8rem !important;
    }

    /* ── Tabs (Navigasi Atas) ── */
    [data-baseweb="tab-list"] { background: transparent !important; gap: 2rem !important; border-bottom: 1px solid var(--border-glass) !important; }
    [data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        color: var(--secondary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        padding: 15px 0 !important;
    }
    [aria-selected="true"][data-baseweb="tab"] {
        border-bottom: 2px solid var(--primary-dark) !important;
    }
    [aria-selected="true"][data-baseweb="tab"] p {
        color: var(--primary) !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)