import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
from datetime import datetime
import time
import gc

st.set_page_config(
    page_title="Analisis Pola Perilaku Konsumen - E-commerce",
    page_icon="",
    layout="wide"
)

BASE_DIR = Path(__file__).parent.parent

MODEL_PATH = BASE_DIR / "models" / "kmeans_best.pkl"
RULE_PATH = BASE_DIR / "processed" / "association_rules.csv"

@st.cache_resource
def load_model():
    for p in [MODEL_PATH, BASE_DIR / "best_kmeans_model.pkl",
              BASE_DIR / "models" / "final" / "kmeans_best.pkl",
              BASE_DIR / "models" / "kmeans_model.pkl"]:
        if p.exists():
            try:
                return joblib.load(p)
            except:
                continue
    return None

@st.cache_data
def load_rules():
    p = RULE_PATH
    if p.exists():
        df = pd.read_csv(p)
        for col in ['antecedents', 'consequents']:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].str.replace(r'frozenset\(\{', '', regex=True)
                df[col] = df[col].str.replace(r'\}', '', regex=True)
                df[col] = df[col].str.replace("'", '', regex=False)
                df[col] = df[col].str.strip()
        for c in ['lift', 'confidence', 'support']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    return None

@st.cache_resource
def load_cluster_info():
    m = load_model()
    if m is None:
        return []
    p = BASE_DIR / "processed" / "data_clean.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    df = df.dropna(subset=['Quantity', 'UnitPrice'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    sample = df.sample(min(50000, len(df)), random_state=42)
    sample['TotalPrice'] = sample['Quantity'] * sample['UnitPrice']
    sample = sample[np.isfinite(sample['TotalPrice'])]
    if sample.empty:
        return []
    preds = m.predict(sample[['Quantity', 'UnitPrice', 'TotalPrice']])
    sample['Cluster'] = preds

    info = []
    order = sample.groupby('Cluster')['TotalPrice'].mean().sort_values()
    for i, (c, _) in enumerate(order.items()):
        data = sample[sample['Cluster'] == c]
        if i == 0:
            label = "Low Spender"
        elif i == len(order) - 1:
            label = "High Spender"
        else:
            label = "Medium Spender"
        info.append({
            'cluster': int(c),
            'label': label,
            'count': len(data),
            'pct': len(data) / len(sample) * 100,
            'avg_qty': data['Quantity'].mean(),
            'avg_price': data['UnitPrice'].mean(),
            'avg_total': data['TotalPrice'].mean(),
            'total_revenue': data['TotalPrice'].sum()
        })
    return info

model = load_model()
rules = load_rules()
cluster_info = load_cluster_info()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Quantity', 'UnitPrice', 'TotalPrice', 'Cluster', 'Segmen'])
if "last_cluster" not in st.session_state:
    st.session_state.last_cluster = None
if "last_segmen" not in st.session_state:
    st.session_state.last_segmen = None
if "save_success" not in st.session_state:
    st.session_state.save_success = False
if "save_timestamp" not in st.session_state:
    st.session_state.save_timestamp = 0
if "process_upload" not in st.session_state:
    st.session_state.process_upload = False

# Reset last_cluster & last_segmen jika history kosong (untuk menghindari muncul tiba-tiba)
if st.session_state.history.empty:
    st.session_state.last_cluster = None
    st.session_state.last_segmen = None

def compute_segment_info(df):
    if df.empty or 'Cluster' not in df.columns:
        return []
    clean = df.copy()
    for c in ['Quantity', 'UnitPrice', 'TotalPrice']:
        clean[c] = pd.to_numeric(clean[c], errors='coerce')
    clean = clean.dropna(subset=['Quantity', 'UnitPrice', 'TotalPrice'])
    clean = clean[np.isfinite(clean['Quantity']) & np.isfinite(clean['UnitPrice']) & np.isfinite(clean['TotalPrice'])]
    if clean.empty:
        return []
    info = []
    order = clean.groupby('Cluster')['TotalPrice'].mean().sort_values()
    for i, (c, _) in enumerate(order.items()):
        data = clean[clean['Cluster'] == c]
        if i == 0:
            label = "Low Spender"
        elif i == len(order) - 1:
            label = "High Spender"
        else:
            label = "Medium Spender"
        info.append({
            'cluster': int(c),
            'label': label,
            'count': len(data),
            'pct': len(data) / len(clean) * 100,
            'avg_qty': data['Quantity'].mean(),
            'avg_price': data['UnitPrice'].mean(),
            'avg_total': data['TotalPrice'].mean(),
            'median_qty': data['Quantity'].median(),
            'median_price': data['UnitPrice'].median(),
            'median_total': data['TotalPrice'].median(),
            'min_qty': data['Quantity'].min(),
            'max_qty': data['Quantity'].max(),
            'min_price': data['UnitPrice'].min(),
            'max_price': data['UnitPrice'].max(),
            'min_total': data['TotalPrice'].min(),
            'max_total': data['TotalPrice'].max(),
            'std_qty': data['Quantity'].std(),
            'std_price': data['UnitPrice'].std(),
            'std_total': data['TotalPrice'].std(),
            'total_revenue': data['TotalPrice'].sum()
        })
    return info

def save_prediction_to_history(qty, price, total, cluster, segmen):
    """Fungsi untuk menyimpan prediksi ke history"""
    total = round(float(total), 2)
    if not np.isfinite(total):
        return
    new_row = pd.DataFrame([{
        'Quantity': int(qty),
        'UnitPrice': round(float(price), 2),
        'TotalPrice': total,
        'Cluster': int(cluster),
        'Segmen': str(segmen)
    }])
    st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
    st.session_state.last_cluster = int(cluster)
    st.session_state.last_segmen = segmen
    st.session_state.save_success = True
    st.session_state.save_timestamp = time.time()

def batch_predict(df, model, batch_size=5000):
    """Prediksi dalam batch untuk menghindari memory overflow"""
    import gc
    
    total_rows = len(df)
    predictions = []
    
    # Clean input data
    batch_data = df[['Quantity', 'UnitPrice', 'TotalPrice']].copy()
    for c in ['Quantity', 'UnitPrice', 'TotalPrice']:
        batch_data[c] = pd.to_numeric(batch_data[c], errors='coerce')
    batch_data = batch_data.dropna()
    batch_data = batch_data[np.isfinite(batch_data['Quantity']) & np.isfinite(batch_data['UnitPrice']) & np.isfinite(batch_data['TotalPrice'])]
    if batch_data.empty:
        return np.array([])
    
    total_rows = len(batch_data)
    
    # Buat placeholder untuk progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total_rows, batch_size):
        batch = batch_data.iloc[i:i+batch_size]
        batch_preds = model.predict(batch)
        predictions.extend(batch_preds)
        
        # Update progress
        progress = min((i + batch_size) / total_rows, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"✓ Memproses: {min(i + batch_size, total_rows)}/{total_rows} baris")
        
        # Memory cleanup setiap batch
        gc.collect()
        
    progress_bar.empty()
    status_text.empty()
    
    return np.array(predictions)

# ======================= SIDEBAR =======================
st.sidebar.markdown(
    "<h2 style='text-align: center;'>  Analisis Pola Perilaku<br>Konsumen</h2>"
    "<p style='text-align: center; color: #888; font-size: 0.85em;'>"
    "Meningkatkan Konversi Pembelian<br>pada Platform E-Commerce</p><hr>",
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "",
    ["  Analisis Segmen", "  Rekomendasi Produk", "  Insight Bisnis", "  Info Model"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Status")
if model:
    st.sidebar.success(f" Model: {model.n_clusters} cluster")
if rules is not None:
    st.sidebar.success(f" Rules: {len(rules)} aturan")
st.sidebar.info(f" Riwayat: {len(st.session_state.history)} analisis")

# Tampilkan notifikasi jika baru saja tersimpan
if st.session_state.save_success and (time.time() - st.session_state.save_timestamp) < 3:
    st.sidebar.success("✓ Data baru disimpan!")

# ======================= ANALISIS SEGMEN =======================
if menu == "  Analisis Segmen":
    st.markdown(
        "<h1 style='text-align: center;'>  Analisis Pola Perilaku Konsumen</h1>"
        "<p style='text-align: center; color: #666;'>Menganalisis pola perilaku konsumen dalam meningkatkan konversi pembelian pada platform E-Commerce</p><hr>",
        unsafe_allow_html=True
    )

    if model is None:
        st.error(" Model K-Means tidak ditemukan. Jalankan training pipeline dulu.")
    else:
        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.markdown("###   Input Cepat")
            st.caption("Isi Quantity dan UnitPrice, hasil langsung muncul.")
            with st.container(border=True):
                qty_fast = st.number_input("Quantity", min_value=1, value=10, key="qty_fast")
                price_fast = st.number_input("UnitPrice (£)", min_value=0.01, value=5.0, step=0.5, key="price_fast")
                total_fast = qty_fast * price_fast
                pred_fast = model.predict([[float(qty_fast), float(price_fast), float(total_fast)]])[0]
                label_map = {c['cluster']: c['label'] for c in cluster_info} if cluster_info else {}
                label_fast = label_map.get(int(pred_fast), f"Cluster {pred_fast}")

            st.markdown(
                f"""
                <div style='padding: 15px; border-radius: 10px; background: #f0f2f6; text-align: center;'>
                    <h3 style='margin: 0; color: #333;'>{label_fast}</h3>
                    <p style='color: #666; margin: 5px 0;'>Cluster {pred_fast}</p>
                    <p style='font-size: 0.85em; color: #888;'>Quantity: {qty_fast} | Harga: £{price_fast:.2f} | Total: £{total_fast:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button("+ Simpan ke Riwayat", use_container_width=True):
                save_prediction_to_history(qty_fast, price_fast, total_fast, pred_fast, label_fast)
                st.success("✓ Data berhasil disimpan ke riwayat!")
                time.sleep(1)
                st.rerun()

        with col_right:
            st.markdown("###   Batch Prediction")
            tab_in, tab_up = st.tabs(["  Input Table", "  Upload File"])

            with tab_in:
                st.caption("Tambah baris sesuai kebutuhan, lalu klik tombol analisis.")
                edited = st.data_editor(
                    pd.DataFrame({'Quantity': pd.Series(dtype='int'), 'UnitPrice': pd.Series(dtype='float')}),
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "Quantity": st.column_config.NumberColumn("Quantity", min_value=1, required=True),
                        "UnitPrice": st.column_config.NumberColumn("UnitPrice (£)", min_value=0.01, format="£%.2f", required=True)
                    }
                )
                if st.button("  Analisis Semua", type="primary", use_container_width=True):
                    if edited.empty:
                        st.warning(" Tabel kosong.")
                    else:
                        try:
                            df = edited.copy()
                            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
                            df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
                            df = df.dropna(subset=['Quantity', 'UnitPrice'])
                            if df.empty:
                                st.warning("Tidak ada data valid setelah filter. Pastikan semua baris terisi dengan benar.")
                                st.stop()
                            df['TotalPrice'] = (df['Quantity'] * df['UnitPrice']).round(2)
                            
                            with st.spinner(" Menganalisis cluster..."):
                                preds = batch_predict(df, model, batch_size=5000)
                            
                            df['Cluster'] = preds
                            label_map = {c['cluster']: c['label'] for c in cluster_info} if cluster_info else {}
                            df['Segmen'] = [label_map.get(int(p), f"Cluster {p}") for p in preds]
                            st.session_state.history = pd.concat([st.session_state.history, df], ignore_index=True)
                            majority = df['Cluster'].mode().iloc[0]
                            st.session_state.last_cluster = int(majority)
                            label_map = {c['cluster']: c['label'] for c in cluster_info} if cluster_info else {}
                            st.session_state.last_segmen = label_map.get(int(majority), f"Cluster {majority}")
                            st.balloons()
                            st.success(f"✓ **{len(df)} transaksi** berhasil dianalisis dan disimpan!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f" Analisis gagal: {e}")

            with tab_up:
                # Set limit untuk analisis upload file
                MAX_ROWS_UPLOAD = 10000
                
                up = st.file_uploader("Upload CSV / Excel (kolom: Quantity, UnitPrice)", type=["csv", "xlsx"])
                if up is not None:
                    try:
                        # Check file size (max 50MB)
                        file_size_mb = up.size / (1024 * 1024)
                        if file_size_mb > 50:
                            st.warning(f" File terlalu besar ({file_size_mb:.1f}MB, max 50MB). Silakan upload file yang lebih kecil.")
                        else:
                            with st.spinner(" Membaca file..."):
                                df = pd.read_excel(up, engine='openpyxl') if up.name.endswith('xlsx') else pd.read_csv(up)
                            
                            # Validasi kolom
                            if not {'Quantity', 'UnitPrice'}.issubset(df.columns):
                                st.error(f" Kolom wajib: Quantity, UnitPrice. Ditemukan: {list(df.columns)}")
                            else:
                                total_rows = len(df)
                                st.info(f" File berhasil dibaca: **{total_rows}** baris")
                                
                                # Data cleaning
                                df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
                                df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
                                df = df.dropna(subset=['Quantity', 'UnitPrice'])
                                df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
                                valid_rows = len(df)
                                
                                if not df.empty:
                                    st.info(f"✓ Setelah filter: **{valid_rows}** baris valid")
                                    
                                    # **CHECK LIMIT - Sangat penting untuk keamanan**
                                    if valid_rows > MAX_ROWS_UPLOAD:
                                        st.warning(
                                            f" **DATA TERLALU BANYAK!**\n\n"
                                            f"File Anda memiliki **{valid_rows:,}** baris valid, "
                                            f"namun sistem hanya bisa memproses maksimal **{MAX_ROWS_UPLOAD:,}** baris untuk keamanan.\n\n"
                                            f"Pilih opsi di bawah:"
                                        )
                                        
                                        col_opt1, col_opt2 = st.columns(2)
                                        with col_opt1:
                                            if st.button("✓ Proses 10,000 Baris Pertama", type="primary", use_container_width=True):
                                                df = df.head(MAX_ROWS_UPLOAD)
                                                st.info(f"✓ Data dipotong menjadi {MAX_ROWS_UPLOAD:,} baris pertama")
                                                st.session_state.process_upload = True
                                        
                                        with col_opt2:
                                            if st.button(" Batal (Upload File Lain)", use_container_width=True):
                                                st.info("Upload dibatalkan. Silakan upload file dengan data yang lebih sedikit.")
                                                st.stop()
                                        
                                        if not st.session_state.get("process_upload", False):
                                            st.stop()
                                    
                                    # Batch prediction dengan progress
                                    st.markdown("**Memproses analisis...**")
                                    label_map = {c['cluster']: c['label'] for c in cluster_info} if cluster_info else {}
                                    df['TotalPrice'] = (df['Quantity'] * df['UnitPrice']).round(2)
                                    
                                    with st.spinner(" Menganalisis cluster..."):
                                        preds = batch_predict(df, model, batch_size=5000)
                                    
                                    df['Cluster'] = preds
                                    df['Segmen'] = [label_map.get(int(p), f"Cluster {p}") for p in preds]
                                    
                                    # Simpan ke history
                                    st.session_state.history = pd.concat([st.session_state.history, df], ignore_index=True)
                                    st.session_state.process_upload = False
                                    
                                    # Update last cluster
                                    majority = df['Cluster'].mode().iloc[0]
                                    st.session_state.last_cluster = int(majority)
                                    label_map = {c['cluster']: c['label'] for c in cluster_info} if cluster_info else {}
                                    st.session_state.last_segmen = label_map.get(int(majority), f"Cluster {majority}")
                                    
                                    st.success(f" **{len(df)} transaksi** berhasil dianalisis dan disimpan ke riwayat!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.warning(" Data kosong setelah filter (semua baris tidak valid).")
                    except Exception as e:
                        st.error(f" Error: {str(e)}")

        st.markdown("---")
        st.markdown("###   Hasil Analisis")

        if st.session_state.history.empty:
            st.info("Belum ada data. Gunakan input cepat, tabel, atau upload file.")
        else:
            tab_hasil, tab_grafik = st.tabs(["  Tabel", "  Grafik"])

            with tab_hasil:
                col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
                with col_h1:
                    st.markdown("**Daftar Hasil Analisis**")
                with col_h2:
                    if st.button("   Hapus Dipilih", type="secondary", use_container_width=True):
                        st.session_state.show_delete_checkboxes = not st.session_state.get("show_delete_checkboxes", False)
                with col_h3:
                    if st.button("   Hapus Semua", type="secondary", use_container_width=True):
                        st.session_state.history = pd.DataFrame(columns=['Quantity', 'UnitPrice', 'TotalPrice', 'Cluster', 'Segmen'])
                        st.session_state.last_cluster = None
                        st.session_state.last_segmen = None
                        st.rerun()
                
                # Initialize checkbox state jika belum ada
                if "delete_checkboxes" not in st.session_state:
                    st.session_state.delete_checkboxes = [False] * len(st.session_state.history)
                
                # Tampilkan checkboxes untuk delete
                if st.session_state.get("show_delete_checkboxes", False):
                    st.markdown("**Pilih baris untuk dihapus:**")
                    cols = st.columns([0.5, 2, 1.5, 1.5, 1.5, 1.5, 1.5])
                    with cols[0]:
                        st.write("** **")
                    with cols[1]:
                        st.write("**Quantity**")
                    with cols[2]:
                        st.write("**UnitPrice**")
                    with cols[3]:
                        st.write("**TotalPrice**")
                    with cols[4]:
                        st.write("**Cluster**")
                    with cols[5]:
                        st.write("**Segmen**")
                    with cols[6]:
                        st.write("")
                    
                    # Resize checkbox list jika jumlah history berubah
                    if len(st.session_state.delete_checkboxes) != len(st.session_state.history):
                        st.session_state.delete_checkboxes = [False] * len(st.session_state.history)
                    
                    for idx, (i, row) in enumerate(st.session_state.history.iterrows()):
                        cols = st.columns([0.5, 2, 1.5, 1.5, 1.5, 1.5, 1.5])
                        with cols[0]:
                            st.session_state.delete_checkboxes[idx] = st.checkbox("", key=f"del_{idx}")
                        with cols[1]:
                            st.write(f"{int(row['Quantity'])}" if pd.notna(row['Quantity']) else "-")
                        with cols[2]:
                            st.write(f"£{row['UnitPrice']:.2f}" if pd.notna(row['UnitPrice']) else "-")
                        with cols[3]:
                            st.write(f"£{row['TotalPrice']:.2f}" if pd.notna(row['TotalPrice']) else "-")
                        with cols[4]:
                            st.write(f"{int(row['Cluster'])}" if pd.notna(row['Cluster']) else "-")
                        with cols[5]:
                            st.write(f"{row['Segmen']}" if pd.notna(row['Segmen']) else "-")
                        with cols[6]:
                            st.write("")
                    
                    col_d1, col_d2 = st.columns([1, 1])
                    with col_d1:
                        if st.button("✓ Hapus Baris Terpilih", use_container_width=True, type="primary"):
                            # Filter dan hapus yang terpilih
                            indices_to_keep = [i for i, selected in enumerate(st.session_state.delete_checkboxes) if not selected]
                            st.session_state.history = st.session_state.history.iloc[indices_to_keep].reset_index(drop=True)
                            st.session_state.delete_checkboxes = [False] * len(st.session_state.history)
                            st.session_state.show_delete_checkboxes = False
                            st.success(f"✓ {len(st.session_state.delete_checkboxes) - len(indices_to_keep)} baris dihapus!")
                            st.rerun()
                    with col_d2:
                        if st.button("✕ Batal", use_container_width=True):
                            st.session_state.delete_checkboxes = [False] * len(st.session_state.history)
                            st.session_state.show_delete_checkboxes = False
                            st.rerun()
                
                # Tampilkan tabel utama
                st.dataframe(st.session_state.history, use_container_width=True)
                
                csv = st.session_state.history.to_csv(index=False).encode('utf-8')
                st.download_button("  Download Semua CSV", csv, "analisis_semua.csv", "text/csv")

            with tab_grafik:
                plot_data = st.session_state.history.copy()
                plot_data = plot_data.dropna(subset=['Segmen'])
                for c in ['Quantity', 'UnitPrice', 'TotalPrice']:
                    plot_data[c] = pd.to_numeric(plot_data[c], errors='coerce')
                plot_data = plot_data.dropna(subset=['Quantity', 'UnitPrice', 'TotalPrice'])
                plot_data = plot_data[np.isfinite(plot_data['TotalPrice'])]

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    if not plot_data.empty:
                        dist = plot_data['Segmen'].value_counts().reset_index()
                        dist.columns = ['Segmen', 'Jumlah']
                        st.plotly_chart(
                            px.pie(dist, values='Jumlah', names='Segmen', title='Distribusi Segmen'),
                            use_container_width=True
                        )
                with col_g2:
                    if not plot_data.empty:
                        avg = plot_data.groupby('Segmen')['TotalPrice'].mean().reset_index()
                        st.plotly_chart(
                            px.bar(avg, x='Segmen', y='TotalPrice', color='Segmen',
                                   title='Rata-rata Total per Segmen', text_auto='.2f'),
                            use_container_width=True
                        )

                if len(plot_data) > 1:
                    st.plotly_chart(
                        px.scatter(plot_data, x='Quantity', y='UnitPrice',
                                   size='TotalPrice', color='Segmen',
                                   title='Sebaran Data Analisis', size_max=20),
                        use_container_width=True
                    )

        # Tampilkan karakteristik HANYA jika ada data analisis yang tersimpan di history
        if not st.session_state.history.empty and st.session_state.last_cluster is not None:
            st.markdown("---")
            st.markdown("###   Karakteristik Segmen Anda")

            # Find cluster info from pre-loaded cluster_info
            ci = None
            for cluster in cluster_info:
                if cluster['cluster'] == st.session_state.last_cluster:
                    ci = cluster
                    break

            if ci is None:
                st.warning(f" **{st.session_state.last_segmen}** (Cluster {st.session_state.last_cluster}) — Model belum memiliki data referensi untuk cluster ini.")
            else:
                color = "#4ECDC4" if ci['label'] == "Low Spender" else "#FF6B6B" if ci['label'] == "High Spender" else "#FFD93D"
                emoji = "" if ci['label'] == "Low Spender" else "" if ci['label'] == "High Spender" else "⭐"
                with st.container(border=True):
                    st.markdown(
                        f"<div style='background:{color}; border-radius:8px; padding:15px 20px; margin-bottom:15px;'>"
                        f"<h3 style='margin:0; color:white;'>{emoji} {ci['label']} — Cluster {ci['cluster']}</h3>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Statistik utama analisis user dari history
                    user_cluster_data = st.session_state.history[
                        st.session_state.history['Cluster'] == st.session_state.last_cluster
                    ].copy()
                    user_cluster_data['TotalPrice'] = pd.to_numeric(user_cluster_data['TotalPrice'], errors='coerce')
                    user_cluster_data = user_cluster_data.dropna(subset=['TotalPrice'])
                    user_cluster_data = user_cluster_data[np.isfinite(user_cluster_data['TotalPrice'])]
                    user_avg_total = user_cluster_data['TotalPrice'].mean() if not user_cluster_data.empty else 0.0
                    user_count = len(user_cluster_data)
                    
                    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
                    with col_k1:
                        st.metric("Rata-rata Belanja Anda", f"£{user_avg_total:.2f}", f"(vs £{ci['avg_total']:.2f} model)")
                    with col_k2:
                        st.metric("Analisis Tersimpan", f"{user_count}", f"dari {ci['count']} referensi")
                    with col_k3:
                        st.metric("Rata-rata Item Model", f"{ci['avg_qty']:.1f}")
                    with col_k4:
                        st.metric("Rata-rata Harga Item", f"£{ci['avg_price']:.2f}")
                    
                    st.markdown("---")
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown(f"** Data Model Referensi**")
                        st.write(f"- Total transaksi: **{ci['count']:,}**")
                        st.write(f"- Total revenue: **£{ci['total_revenue']:,.0f}**")
                        st.write(f"- Persentase: **{ci['pct']:.1f}%** dari total dataset")
                    with col_r2:
                        st.markdown(f"** Analisis Anda saat ini**")
                        st.write(f"- Transaksi tersimpan: **{user_count}**")
                        st.write(f"- Rata-rata belanja: **£{user_avg_total:.2f}**")
                        st.write(f"- Total belanja: **£{user_cluster_data['TotalPrice'].sum():,.2f}**")

# ======================= REKOMENDASI PRODUK =======================
elif menu == "  Rekomendasi Produk":
    st.markdown(
        "<h1 style='text-align: center;'>  Rekomendasi Produk</h1>"
        "<p style='text-align: center; color: #666;'>Cari produk untuk mengetahui produk lain yang sering dibeli bersamaan</p><hr>",
        unsafe_allow_html=True
    )

    if rules is None or rules.empty:
        st.error(" Association rules tidak ditemukan.")
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_lift = st.slider("Minimum Lift", 1.0, 25.0, 3.0, 0.5)
        with col_f2:
            min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.3, 0.05)

        filtered = rules[(rules['lift'] >= min_lift) & (rules['confidence'] >= min_conf)]

        all_products = sorted(set(
            p for p in pd.concat([
                filtered['antecedents'].dropna().astype(str).str.strip(),
                filtered['consequents'].dropna().astype(str).str.strip()
            ]).unique() if p.lower() not in ('', 'nan')
        ))

        produk = st.selectbox("Pilih produk:", [""] + all_products)
        if produk:
            match = filtered[
                filtered['antecedents'].str.contains(produk, case=False, na=False, regex=False) |
                filtered['consequents'].str.contains(produk, case=False, na=False, regex=False)
            ].sort_values('lift', ascending=False)

            if match.empty:
                st.info(f"Tidak ada rekomendasi untuk '{produk}' dengan filter saat ini.")
            else:
                st.success(f"  {len(match)} rekomendasi ditemukan")
                st.dataframe(
                    match[['antecedents', 'consequents', 'confidence', 'lift', 'support']].head(10),
                    use_container_width=True,
                    column_config={
                        "confidence": st.column_config.NumberColumn("Confidence", format="%.1%"),
                        "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
                        "support": st.column_config.NumberColumn("Support", format="%.1%")
                    }
                )
                for _, r in match.head(5).iterrows():
                    with st.container(border=True):
                        col_a, col_b, col_c = st.columns([2, 2, 1])
                        with col_a:
                            st.markdown(f"** Jika beli:**<br>{r['antecedents']}", unsafe_allow_html=True)
                        with col_b:
                            st.markdown(f"** Maka beli juga:**<br>{r['consequents']}", unsafe_allow_html=True)
                        with col_c:
                            st.metric("Lift", f"{float(r['lift']):.1f}x")
                        st.progress(min(float(r['confidence']), 1.0))
                        st.caption(f"Confidence: {float(r['confidence']):.1%} | Support: {float(r['support']):.1%}")

        st.markdown("---")
        st.markdown("###   Top Association Rules")
        st.dataframe(
            rules.nlargest(20, 'lift')[['antecedents', 'consequents', 'confidence', 'lift', 'support']],
            use_container_width=True,
            column_config={
                "confidence": st.column_config.NumberColumn("Confidence", format="%.1%"),
                "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
                "support": st.column_config.NumberColumn("Support", format="%.1%")
            }
        )

# ======================= INSIGHT BISNIS =======================
elif menu == "  Insight Bisnis":
    st.markdown(
        "<h1 style='text-align: center;'>  Insight Bisnis</h1>"
        "<p style='text-align: center; color: #666;'>Strategi meningkatkan konversi berdasarkan hasil analisis</p><hr>",
        unsafe_allow_html=True
    )

    if st.session_state.last_cluster is not None:
        cluster_data = st.session_state.history[st.session_state.history['Cluster'] == st.session_state.last_cluster]
        ci = None
        if not cluster_data.empty:
            result = compute_segment_info(cluster_data)
            ci = result[0] if result else None
        if ci is None:
            st.info(f"Segmen Anda: **{st.session_state.last_segmen}** (Cluster {st.session_state.last_cluster}). Simpan hasil analisis ke riwayat untuk melihat strategi.")
        else:
            st.markdown("###   Strategi untuk Segmen Anda")
            if ci['label'] == "Low Spender":
                icon, strategi, target = " ", "Bundling diskon, promosi quantity-based, gratis ongkir minimal belanja", "Tingkatkan frekuensi & AOV"
            elif ci['label'] == "High Spender":
                icon, strategi, target = " ", "Program VIP, akses early bird, produk eksklusif, personal shopper", "Maximalkan retensi & loyalitas"
            else:
                icon, strategi, target = " ", "Personalisasi rekomendasi, upsell medium-to-premium, tiered rewards", "Konversi ke High Spender"

            with st.container(border=True):
                st.markdown(f"#### {icon} {ci['label']} — {ci['count']} transaksi Anda (Cluster {ci['cluster']})")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Rata-rata Belanja", f"£{ci['avg_total']:.2f}", f"±£{ci['std_total']:.2f}")
                with col_s2:
                    st.metric("Median Belanja", f"£{ci['median_total']:.2f}")
                with col_s3:
                    st.metric("Rata-rata Jumlah Item", f"{ci['avg_qty']:.1f}", f"±{ci['std_qty']:.1f}")
                with col_s4:
                    st.metric("Total Revenue", f"£{ci['total_revenue']:,.0f}")
                st.info(f"**Strategi:** {strategi}")
                st.info(f"**Target:** {target}")

    if rules is not None and not rules.empty:
        st.markdown("###   Rekomendasi Bundle Produk")
        top_bundles = rules.nlargest(10, 'lift')
        for _, r in top_bundles.iterrows():
            with st.container(border=True):
                st.markdown(f"**{r['antecedents']}** + **{r['consequents']}**")
                st.write(f"Lift: {float(r['lift']):.2f}x | Confidence: {float(r['confidence']):.1%}")

    if st.session_state.last_cluster is not None and rules is not None and not rules.empty:
        cluster_data = st.session_state.history[st.session_state.history['Cluster'] == st.session_state.last_cluster]
        ci = None
        if not cluster_data.empty:
            result = compute_segment_info(cluster_data)
            ci = result[0] if result else None

        if ci:
            st.markdown("###   Rekomendasi Prioritas")
            high_data = ci if ci['label'] == "High Spender" else None
            low_data = ci if ci['label'] == "Low Spender" else None

            top_rule = rules.nlargest(1, 'lift').iloc[0]
            reco_list = []

            if high_data:
                reco_list.append(f"**Fokus pada {high_data['label']}** — {high_data['pct']:.1f}% data Anda dengan rata-rata belanja £{high_data['avg_total']:.2f}. "
                                 f"Beri program loyalitas & akses eksklusif untuk pertahankan segmen ini.")
            if low_data:
                reco_list.append(f"**Tingkatkan {low_data['label']}** — {low_data['pct']:.1f}% data Anda dengan rata-rata belanja £{low_data['avg_total']:.2f}. "
                                 f"Terapkan bundling diskon & gratis ongkir untuk dorong AOV.")
            reco_list.append(f"**Bundle produk unggulan:** Pelanggan yang beli **{top_rule['antecedents']}** cenderung juga beli "
                             f"**{top_rule['consequents']}** (Lift: {float(top_rule['lift']):.1f}x). Jadikan paket bundling.")
            mid_rules = rules[(rules['lift'] >= rules['lift'].median()) & (rules['confidence'] >= 0.5)]
            if len(mid_rules) > 0:
                reco_list.append(f"**{len(mid_rules)} aturan asosiasi** memiliki confidence ≥ 50% — "
                                 f"prioritaskan untuk cross-selling di halaman checkout & rekomendasi produk.")
            for r in reco_list:
                st.success(r)

# ======================= INFO MODEL =======================
elif menu == "  Info Model":
    st.markdown(
        "<h1 style='text-align: center;'>  Informasi Model</h1><hr>",
        unsafe_allow_html=True
    )

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown("###   K-Means Clustering")
        if model:
            st.write(f"- **Jumlah Cluster:** {model.n_clusters}")
            st.write(f"- **Inertia:** {model.inertia_:.2f}")
            st.write(f"- **Iterasi:** {model.n_iter_}")
            st.write(f"- **Fitur:** Quantity, UnitPrice, TotalPrice")
            st.write(f"- **File:** `{MODEL_PATH.name}`")

    with col_i2:
        st.markdown("###   Association Rules")
        if rules is not None and not rules.empty:
            st.write(f"- **Total Rules:** {len(rules)}")
            st.write(f"- **Lift Min:** {rules['lift'].min():.2f}")
            st.write(f"- **Lift Max:** {rules['lift'].max():.2f}")
            st.write(f"- **Lift Avg:** {rules['lift'].mean():.2f}")
            st.write(f"- **Confidence Avg:** {rules['confidence'].mean():.1%}")
            st.write(f"- **File:** `association_rules.csv`")

    st.markdown("---")
    st.markdown("###   Dataset")
    st.write("**Online Retail Dataset** — UCI Machine Learning Repository")
    st.write("- Transaksi e-commerce UK (2010—2011)")
    st.write("- 541.909 transaksi, 8 kolom")
    st.write("- Preprocessing: hapus missing values, cancel, return, harga invalid")

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888; font-size: 0.8em;'>"
        f"Aplikasi Analisis Pola Perilaku Konsumen v2.0 | {datetime.now().strftime('%Y-%m-%d')}</p>",
        unsafe_allow_html=True
    )
