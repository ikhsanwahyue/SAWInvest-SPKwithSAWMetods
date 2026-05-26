
import streamlit as st          
import pandas as pd             
import numpy as np              
import matplotlib.pyplot as plt 

# --- KONFIGURASI HALAMAN ---
# Mengatur layout web agar tampil lebar (wide) dan sidebar terbuka secara default
st.set_page_config(page_title="SPK SMART SWING", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Menggunakan Markdown untuk injeksi CSS guna mengubah tema aplikasi menjadi Dark Mode estetik
st.markdown("""
<style>
    :root {
        --primary-color: #38BDF8 !important;
        --background-color: #0F172A !important;
        --secondary-background-color: #1E293B !important;
        --text-color: #F8FAFC !important;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] { 
        background-color: #0F172A !important; 
        border-right: 1px solid #1E293B; 
    }
    
    p, span, li, label, td, th, small, [data-testid="stMarkdownContainer"] p {
        color: #CBD5E1 !important;
    }
    h1, h2, h3, h4, h5, h6, strong {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stSlider div[data-baseweb="slider"] > div > div > div:first-child {
        background-color: #38BDF8 !important; /* Warna jalur slider yang terisi */
    }
    .stSlider div[data-baseweb="slider"] [role="slider"] {
        background-color: #38BDF8 !important; /* Warna buletan (thumb) slider */
        border: 2px solid #F8FAFC !important;
        box-shadow: 0 0 5px rgba(56, 189, 248, 0.5) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: #0F172A !important; padding-top: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: transparent !important; color: #94A3B8 !important;
        font-weight: 600; border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #F8FAFC !important; border-bottom: 2px solid #38BDF8 !important;
    }
    .stTabs [aria-selected="true"] p { color: #F8FAFC !important; }
    
    [data-testid="stMetric"], [data-testid="stExpander"] {
        background-color: #1E293B !important; border: 1px solid #334155 !important;
        border-radius: 12px !important; padding: 15px 20px !important;
    }
    [data-testid="stMetricLabel"] p { color: #38BDF8 !important; font-weight: 600; font-size: 12px; }
    [data-testid="stMetricValue"] { color: #F8FAFC !important; font-size: 32px !important; font-weight: 700; }

    div[data-testid="stDataFrame"] {
        background-color: #1E293B !important; border-radius: 12px; border: 1px solid #334155; padding: 5px;
    }
    
    .stButton>button {
        background-color: #1E293B !important; color: #F8FAFC !important;            
        border-radius: 8px !important; font-weight: 600 !important;
        border: 1px solid #38BDF8 !important; width: 100%; transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #38BDF8 !important; color: #0F172A !important; box-shadow: 0 0 10px #38BDF8;          
    }
    
    .stAlert { background-color: #1E293B !important; border: 1px solid #334155 !important; }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data  # Dekorator untuk menyimpan data di memori agar loading lebih cepat
def load_data():
    try:
        # Membaca file CSV yang berisi daftar saham
        df = pd.read_csv('data/DaftarSaham.csv')
        return df
    except FileNotFoundError:
        st.error("File 'data/DaftarSaham.csv' tidak ditemukan! Pastikan struktur folder sudah benar.")
        st.stop()

# Preprocessing Data
def preprocess_data(df):

    # Menghapus baris yang memiliki nilai kosong pada kolom-kolom penting untuk analisis
    df_clean = df.dropna(subset=['Code', 'ListingDate', 'Shares', 'ListingBoard', 'LastPrice', 'MarketCap']).copy()

    # Ekstraksi kolom untuk dijadikan parameter SAW
    c1 = df_clean['MarketCap'].values
    c2 = df_clean['LastPrice'].values
    c3 = df_clean['Shares'].values

    # Mapping kriteria kualitatif (Papan Bursa) ke bentuk angka (kuantitatif)
    board_mapping = {'Utama': 3, 'Pengembangan': 2}
    c4 = df_clean['ListingBoard'].map(lambda x: board_mapping.get(x, 1)).values     
    
    # Menghitung umur emiten berdasarkan tahun listing
    df_clean['ListingYear'] = pd.to_datetime(df_clean['ListingDate'], errors='coerce').dt.year
    df_clean['ListingYear'] = df_clean['ListingYear'].fillna(2026)
    c5 = (2026 - df_clean['ListingYear']).values
    
    # Menggabungkan semua kriteria ke dalam satu matriks (Vektorisasi NumPy)
    X = np.column_stack((c1, c2, c3, c4, c5))
    return df_clean, X

# --- ALGORITMA SAW (Simple Additive Weighting) ---
def calculate_saw(X, weights):

    # Memisahkan matriks X menjadi masing-masing kriteria
    c1, c2, c3, c4, c5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    
    # Normalisasi Matriks (Benefit: Data/Max, Cost: Min/Data)
    max_c1, max_c3, max_c4, max_c5 = np.max(c1), np.max(c3), np.max(c4), np.max(c5)
    min_c2 = np.min(c2[c2 > 0]) if np.any(c2 > 0) else 1e-9 
    
    # Menggabungkan hasil normalisasi menjadi matriks R
    r1 = np.where(max_c1 > 0, c1 / max_c1, 0) 
    r2 = np.where(c2 > 0, min_c2 / c2, 0)     
    r3 = np.where(max_c3 > 0, c3 / max_c3, 0)
    r4 = np.where(max_c4 > 0, c4 / max_c4, 0)
    r5 = np.where(max_c5 > 0, c5 / max_c5, 0)
    
    R = np.column_stack((r1, r2, r3, r4, r5)) 
    W = np.array(weights)    # Mengubah daftar bobot jadi array NumPy                 
    
    # Perhitungan Skor Preferensi (V = Sum(Normalisasi * Bobot))
    V = np.sum(R * W, axis=1)
    return R, V

# Fungsi Main
def main():
    df_raw = load_data()
    df, X = preprocess_data(df_raw.copy())
    
    # Sidebar untuk pengaturan bobot kriteria SAW
    st.sidebar.title("⚙️ Pengaturan Bobot SAW")
    style = st.sidebar.selectbox(
        "Pilih Profil Investasi:",
        ["Swing Trade Agresif (High Risk)", "Swing Trade Konservatif (Low Risk)", "Custom Bobot (Manual)"]
    )
    
    # Logika penentuan bobot berdasarkan pilihan profil investasi
    if style == "Swing Trade Agresif (High Risk)":
        weights = [0.15, 0.30, 0.35, 0.10, 0.10]
        st.sidebar.info("- Mode Agresif Aktif. Fokus tinggi pada harga murah dan likuiditas (volatilitas).")
        w1, w2, w3, w4, w5 = weights
    elif style == "Swing Trade Konservatif (Low Risk)":
        weights = [0.35, 0.10, 0.15, 0.20, 0.20]
        st.sidebar.info("- Mode Konservatif Aktif. Fokus pada Market Cap raksasa (Bluechip) dan emiten mapan.")
        w1, w2, w3, w4, w5 = weights
    else:

        # Input custom
        st.sidebar.markdown("### Penyesuaian Bobot Kriteria")

        w1 = st.sidebar.slider("C1: Market Capitalization (Benefit)", 0.0, 1.0, 0.20, 0.05)
        w2 = st.sidebar.slider("C2: Last Price (Cost)", 0.0, 1.0, 0.20, 0.05)
        w3 = st.sidebar.slider("C3: Shares Outstanding (Benefit)", 0.0, 1.0, 0.20, 0.05)
        w4 = st.sidebar.slider("C4: Listing Board (Benefit)", 0.0, 1.0, 0.20, 0.05)
        w5 = st.sidebar.slider("C5: Age on Bourse (Benefit)", 0.0, 1.0, 0.20, 0.05)
        weights = [w1, w2, w3, w4, w5]
    
    # Validasi total bobot harus 1.0 untuk memastikan sistem SAW berjalan dengan benar
    total_weight = sum(weights)
    is_valid = np.isclose(total_weight, 1.0)
    
    if is_valid:
        st.sidebar.success(f"Total Bobot Saat Ini: {total_weight:.2f} (Sistem Valid)")
    else:
        st.sidebar.error(f"Total Bobot Saat Ini: {total_weight:.2f} (TIDAK VALID: Wajib 1.00)")
        
    # Eksekusi perhitungan SAW hanya jika bobot valid untuk menghindari hasil yang tidak akurat
    R, V = calculate_saw(X, weights)
        
    # Navigasi Tab
    tab1, tab2, tab3, tab4 = st.tabs([
        "Market Intelligence", 
        "Quant Ranking Engine", 
        "Visualisasi Performa", 
        "Profil Pengembang"
    ])
    
    # Tab 1
    with tab1:
        st.title("SAW-Invest : Sistem Pendukung Keputusan Seleksi Emiten")
        st.markdown("Platform **Sistem Pendukung Keputusan (SPK)** berbasis komputasi linier **Simple Additive Weighting (SAW)**. Membantu investor menyeleksi peluang *Swing Trading* optimal secara objektif, logis, dan terukur secara matematis.")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Populasi Data", len(df_raw))
        col2.metric("Data Bersih (Lolos Seleksi)", len(df))
        col3.metric("Dimensi Matriks Kriteria", 5)
        
        st.subheader("Dataset Bursa Fundamental")
        st.dataframe(df_raw, use_container_width=True)
        
    # Tab 2
    with tab2:
        st.title("Mesin Perangkingan & Trading Plan")
        
        st.markdown("#### Matriks Normalisasi (R) - Evaluasi Lintas Dimensi")
        df_R = pd.DataFrame(R, columns=["C1 (MarketCap)", "C2 (Price)", "C3 (Shares)", "C4 (Board)", "C5 (Age)"])
        df_R.insert(0, "Code", df["Code"].values)
        st.dataframe(df_R, use_container_width=True)
        
        btn_calc = st.button("Jalankan Algoritma Perangkingan", disabled=not is_valid, use_container_width=True)
        
        if btn_calc:
            ranked_indices = np.argsort(V)[::-1]
            top_10_indices = ranked_indices[:10]
            
            df_top10 = df.iloc[top_10_indices].copy()
            df_top10['Skor Preferensi (V)'] = V[top_10_indices]
            df_top10['Peringkat'] = range(1, 11)
            df_top10['Harga Entry / Masuk'] = df_top10['LastPrice']
            
            if "Agresif" in style:
                tp_pct, sl_pct = 0.15, -0.07 
            elif "Konservatif" in style:
                tp_pct, sl_pct = 0.08, -0.04 
            else: 
                tp_pct, sl_pct = 0.10, -0.05
                
            df_top10['Take Profit (TP)'] = np.round(df_top10['Harga Entry / Masuk'] * (1 + tp_pct)).astype(int)
            df_top10['Stop Loss (SL)'] = np.round(df_top10['Harga Entry / Masuk'] * (1 + sl_pct)).astype(int)
            
            df_display = df_top10[['Peringkat', 'Code', 'Name', 'Sector', 'Skor Preferensi (V)', 'Harga Entry / Masuk', 'Take Profit (TP)', 'Stop Loss (SL)']]
            
            st.subheader("Rekomendasi Top 10 Emiten Terbaik")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            st.session_state['df_top10'] = df_top10
            st.session_state['df_all'] = df
            st.session_state['is_calculated'] = True

    # Tab 3   
    with tab3:
        st.title("Visualisasi Data Komparatif (Matplotlib)")
        
        if not st.session_state.get('is_calculated', False):
            st.info("Tekan tombol 'Jalankan Algoritma Perangkingan' pada Tab sebelumnya untuk merender grafik.")
        else:
            df_top10 = st.session_state['df_top10']
            df_all = st.session_state['df_all']

            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                st.subheader("Distribusi Skor Preferensi")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                fig1.patch.set_facecolor('#1E293B')
                ax1.set_facecolor('#1E293B')
                
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.spines['bottom'].set_color('#334155')
                ax1.spines['left'].set_color('#334155')
                
                colors_bar = plt.cm.Blues(np.linspace(0.5, 1, len(df_top10)))
                bars = ax1.bar(df_top10['Code'], df_top10['Skor Preferensi (V)'], color=colors_bar, edgecolor='#38BDF8')
                
                ax1.set_ylabel("Nilai Preferensi (V)", color='#94A3B8')
                ax1.tick_params(colors='#CBD5E1')
                plt.xticks(rotation=45)
                ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#94A3B8')
                st.pyplot(fig1)
                
            with col_v2:
                st.subheader("Diversifikasi Sektor Industri")
                sector_counts = df_top10['Sector'].value_counts()
                
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                fig2.patch.set_facecolor('#1E293B')
                
                explode = [0.1 if i == 0 else 0 for i in range(len(sector_counts))]
                colors_pie = plt.cm.GnBu(np.linspace(0.4, 0.9, len(sector_counts)))
                
                ax2.pie(
                    sector_counts.values, 
                    labels=sector_counts.index, 
                    autopct='%1.1f%%', 
                    startangle=140, 
                    colors=colors_pie,
                    explode=explode,
                    shadow=True,
                    textprops={'color':"w", 'weight':'bold'}
                )
                st.pyplot(fig2)
                
            st.subheader("Sebaran Populasi Saham Global")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            fig3.patch.set_facecolor('#1E293B')
            ax3.set_facecolor('#0F172A')
            
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['bottom'].set_color('#334155')
            ax3.spines['left'].set_color('#334155')
            
            ax3.scatter(df_all['LastPrice'], df_all['MarketCap'], alpha=0.6, color='#38BDF8', edgecolor='white', linewidth=0.5)
            ax3.set_xlabel("Harga Saham (Rp) - Skala Log", color='#94A3B8', fontweight='bold')
            ax3.set_ylabel("Kapitalisasi Pasar (Rp) - Skala Log", color='#94A3B8', fontweight='bold')
            ax3.tick_params(colors='#CBD5E1')
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            ax3.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.2, color='white')
            st.pyplot(fig3)

    # Tab 4 
    with tab4:
        st.title("Identitas Pengembang")
        
        st.markdown('''
        **Tim Pengembang**
        
        * 👤 Ikhsan Wahyu Endriarto  [123240241]
        * 👤 Aziz Nabil Putra Darmawan  [123240239]
        
        **Detail Akademik**
                    
        * Universitas : Universitas Pembangunan Nasional "Veteran" Yogyakarta
        * Fakultas    : Fakultas Teknik Industri (FTI)
        * Jurusan     : Informatika 
        * Prodi       : S1 Informatika
        * Mata Kuliah : Sistem Pendukung Keputusan (SCPK)
        ''')

if __name__ == '__main__':
    main()