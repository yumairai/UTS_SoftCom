import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from core.backend import trimf, U_PM25, U_PM10, U_CO, U_OUT, MF_OUT_DEFAULT

COLOR_LOW  = '#a3cce4' # Light Blue
COLOR_MID  = '#ffb4a3' # Peach/Primary
COLOR_HIGH = '#ee6c4d' # Orange/Danger
COLORS_MF  = [COLOR_LOW, COLOR_MID, COLOR_HIGH]

COLOR_MAP = {
    'Sangat Aman': '#10B981', # Green
    'Aman': '#a3cce4',        # Light Blue
    'Tidak Sehat': '#ffb4a3', # Peach
    'Berbahaya': '#ee6c4d'    # Orange
}

def trimf(x, abc):
    a, b, c = abc
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def plot_membership_functions(mf_pm25, mf_pm10, mf_co, val25, val10, valco, score):
    bg_color = '#0a1421' 
    text_color = '#92bbd3'
    
    spine_color = (166/255, 138/255, 132/255, 0.3)
    legend_bg_color = (44/255, 53/255, 68/255, 0.8)
    
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(bg_color)
    
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.3], hspace=0.55, wspace=0.8)
    
    ax1 = fig.add_subplot(gs[0, 0:2]) 
    ax2 = fig.add_subplot(gs[0, 2:4]) 
    ax3 = fig.add_subplot(gs[0, 4:6]) 
    ax4 = fig.add_subplot(gs[1, 1:5]) 
    
    axes = [ax1, ax2, ax3, ax4]
    
    colors_out = [COLOR_MAP['Sangat Aman'], COLOR_MAP['Aman'], COLOR_MAP['Tidak Sehat'], COLOR_MAP['Berbahaya']]
    
    datasets = [
        (np.arange(0, 301, 1), mf_pm25, val25, 'INPUT: PM2.5 (µg/m³)', COLORS_MF),
        (np.arange(0, 201, 1), mf_pm10, val10, 'INPUT: PM10 (µg/m³)', COLORS_MF),
        (np.arange(0, 51, 0.5), mf_co, valco, 'INPUT: CO (ppm)', COLORS_MF),
        (U_OUT, MF_OUT_DEFAULT, score, '🎯 OUTPUT: DEFUZZIFIKASI ISPU SCORE', colors_out)
    ]
    
    for ax, (universe, mf_params, val, title, colors) in zip(axes, datasets):
        ax.set_facecolor(bg_color)
        ax.spines['bottom'].set_color(spine_color)
        ax.spines['left'].set_color(spine_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors=text_color)
        
        is_output = "OUTPUT" in title
        title_color = '#ee6c4d' if is_output else '#dae3f7'
        title_size = 14 if is_output else 11
        
        ax.set_title(title, color=title_color, fontweight='bold', fontfamily='sans-serif', fontsize=title_size, pad=15)
        ax.set_ylim(-0.05, 1.25)
        
        for (name, params), color in zip(mf_params.items(), colors):
            y_vals = [trimf(v, params) for v in universe]
            line_w = 3.0 if is_output else 2.0
            ax.plot(universe, y_vals, color=color, linewidth=line_w, label=name.upper())
            ax.fill_between(universe, y_vals, alpha=0.15, color=color)
            
        ax.axvline(val, color='#ffb4a3', linewidth=3, linestyle='--', label=f'NILAI: {val:.1f}')
        ax.legend(facecolor=legend_bg_color, edgecolor=spine_color, labelcolor='#dae3f7', fontsize=9, loc='upper right')
        
    return fig

def plot_ann_probabilities(proba, classes, label_pred):
    bg_color = '#0a1421'
    colors = [COLOR_MAP.get(c, '#ffffff') for c in classes]
    
    fig = go.Figure(go.Bar(
        x=proba,
        y=classes,
        orientation='h',
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in proba],
        textposition='outside',
        textfont=dict(color='#dae3f7', size=14, family='sans-serif')
    ))
    
    fig.update_layout(
        title=dict(text='🎯 OUTPUT LAYER: PROBABILITAS KLASIFIKASI', font=dict(color='#ee6c4d', size=16), x=0.5, xanchor='center'),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        xaxis=dict(range=[0, 1.2], showgrid=True, gridcolor='rgba(166, 138, 132, 0.15)', tickfont=dict(color='#92bbd3')),
        yaxis=dict(autorange="reversed", tickfont=dict(color='#dae3f7', size=14)),
        margin=dict(l=20, r=20, t=60, b=20),
        height=280
    )
    return fig

def plot_confusion_matrix(classes):
    """Membuat visualisasi Confusion Matrix statis evaluasi ANN"""
    z = [[125, 3, 0, 0],    
         [5, 480, 12, 0],   
         [0, 18, 195, 4],   
         [0, 0, 6, 68]]     
         
    z = z[::-1]
    y_labels = classes[::-1]
    
    colorscale = [[0.0, '#0a1421'], [0.2, '#131c2a'], [0.5, '#a3cce4'], [0.8, '#ffb4a3'], [1.0, '#ee6c4d']]

    fig = ff.create_annotated_heatmap(
        z, x=classes, y=y_labels, 
        colorscale=colorscale,
        showscale=True,
        font_colors=['#dae3f7', '#0a1421'] 
    )
    
    # PERBAIKAN: Memindahkan sumbu X ke bawah dan menambah margin atas
    fig.update_layout(
        title=dict(
            text='EVALUASI DATA UJI: CONFUSION MATRIX (ANN)', 
            font=dict(color='#a3cce4', size=14), 
            x=0.5, 
            xanchor='center',
            y=0.95 # Mendorong judul sedikit lebih ke atas
        ),
        plot_bgcolor='#0a1421',
        paper_bgcolor='#0a1421',
        xaxis=dict(
            side='bottom', # <--- Memaksa sumbu X berada di bawah
            title=dict(text='Kelas Prediksi (Predicted)', font=dict(color='#92bbd3')), 
            tickfont=dict(color='#dae3f7')
        ),
        yaxis=dict(
            title=dict(text='Kelas Aktual (True)', font=dict(color='#92bbd3')), 
            tickfont=dict(color='#dae3f7')
        ),
        margin=dict(l=40, r=20, t=80, b=60), # <--- Margin (t)op dan (b)ottom diperbesar
        height=350 # Ditinggikan sedikit agar tidak terlalu sesak
    )
    return fig