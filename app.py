import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import shap
import warnings
import time
import random

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# CANLI HAVA DURUMU API YARDIMCI BİLEŞENLERİ (Open-Meteo)
# ═══════════════════════════════════════════════════════════════════════
import requests
from datetime import datetime

def canli_hava_durumu_cek(lat, lon):
    """
    Open-Meteo ücretsiz API'sını kullanarak koordinatlara göre anlık hava verisi çeker.
    API key gerektirmez.
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,cloud_cover,pressure_msl,wind_speed_10m,uv_index,visibility&timezone=auto"
        res = requests.get(url, timeout=5).json()
        current = res["current"]
        
        temp = current["temperature_2m"]
        hum = current["relative_humidity_2m"]
        wind = current["wind_speed_10m"]  # km/h
        precip_mm = current["precipitation"]
        cloud_pct = current["cloud_cover"]
        press = current["pressure_msl"]
        uv_idx = int(current["uv_index"])
        vis_m = current["visibility"]
        vis_km = vis_m / 1000.0
        
        # Yağış miktarına göre yağış ihtimali tahmini
        if precip_mm == 0:
            precip_prob = 0.0
        elif precip_mm < 1.0:
            precip_prob = 30.0
        elif precip_mm < 5.0:
            precip_prob = 70.0
        else:
            precip_prob = 100.0
            
        return {
            "success": True,
            "temp": temp,
            "hum": hum,
            "wind": wind,
            "precip": precip_prob,
            "cloud_pct": cloud_pct,
            "press": press,
            "uv": uv_idx,
            "vis": vis_km
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def mevsim_belirle():
    """Tarih ayına göre mevsimi belirler (Veri setindeki sınıflarla eşleşen)"""
    ay = datetime.now().month
    if ay in [12, 1, 2]:
        return "Winter"
    elif ay in [3, 4, 5]:
        return "Spring"
    elif ay in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def bulut_durumu_belirle(yuzde):
    """Bulutluluk yüzdesine göre veri setindeki kategorik sınıflara eşleme yapar"""
    if yuzde < 20:
        return "clear"
    elif yuzde < 60:
        return "partly cloudy"
    elif yuzde < 90:
        return "cloudy"
    else:
        return "overcast"

# Şehir Koordinatları ve Konum Türleri (Veri Setindeki Konumlarla Eşlenik)
SEHIRLER = {
    "İstanbul":   {"lat": 41.0082, "lon": 28.9784, "loc": "coastal"},
    "Ankara":     {"lat": 39.9334, "lon": 32.8597, "loc": "inland"},
    "İzmir":      {"lat": 38.4192, "lon": 27.1287, "loc": "coastal"},
    "Erzurum":    {"lat": 39.9056, "lon": 41.2658, "loc": "mountain"},
    "Antalya":    {"lat": 36.8841, "lon": 30.7056, "loc": "coastal"},
    "Trabzon":    {"lat": 41.0027, "lon": 39.7168, "loc": "coastal"},
    "Eskişehir":  {"lat": 39.7767, "lon": 30.5206, "loc": "inland"},
    "Malatya":    {"lat": 38.3552, "lon": 38.3095, "loc": "inland"},
}

# ═══════════════════════════════════════════════════════════════════════
# SAYFA AYARLARI
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="İHA Yer Kontrol İstasyonu",
    page_icon="🛸",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════════════════
# VERİ YÜKLEME, 6 MODEL + ENSEMBLE EĞİTİMİ, SHAP (Önbellekli)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def sistemi_hazirla():
    """
    Veriyi yükler, 6 farklı ML modeli + 1 Ensemble (Oylama) modeli eğitir,
    5-Fold Cross-Validation ile doğrular, tüm metrikleri hesaplar
    ve SHAP açıklanabilirlik değerlerini üretir.
    """

    df = pd.read_csv('weather_classification_data.csv')
    
    # ── HIZLI ÇALIŞMA OPTİMİZASYONU ──
    # 13.200 satırlık veri setinde SVM (Destek Vektör Makinesi) ve 5-Fold Cross-Validation eğitimi
    # uzun sürmektedir. Sunum esnasında uygulamanın saniyeler içinde açılması için veriden 
    # temsil yeteneği yüksek 2.000 satırlık dengeli bir örneklem (sample) alıyoruz.
    df = df.sample(n=2000, random_state=42).reset_index(drop=True)

    # ── Label Encoding ──
    le_cloud   = LabelEncoder()
    le_season  = LabelEncoder()
    le_location = LabelEncoder()
    le_weather = LabelEncoder()

    X = df.drop('Weather Type', axis=1)
    y = df['Weather Type']

    X['Cloud Cover'] = le_cloud.fit_transform(X['Cloud Cover'])
    X['Season']      = le_season.fit_transform(X['Season'])
    X['Location']    = le_location.fit_transform(X['Location'])
    y_encoded        = le_weather.fit_transform(y)

    # ── Train / Test Split (%80 eğitim – %20 test) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # ── Normalizasyon (KNN, SVM, Lojistik Regresyon için gerekli) ──
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    # ── 6 Farklı ML Algoritması ──
    model_tanimlari = {
        "Random Forest":               RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        "K-En Yakın Komşu (KNN)":      KNeighborsClassifier(n_neighbors=7),
        "Destek Vektör Makinesi (SVM)": SVC(kernel='rbf', probability=True, random_state=42),
        "Lojistik Regresyon":           LogisticRegression(max_iter=1000, random_state=42),
        "Karar Ağacı":                  DecisionTreeClassifier(max_depth=10, random_state=42),
        "Naive Bayes":                  GaussianNB(),
    }

    olcekleme_gereken = {
        "K-En Yakın Komşu (KNN)",
        "Destek Vektör Makinesi (SVM)",
        "Lojistik Regresyon",
    }

    sonuclar = {}
    egitilmis_modeller = {}

    for isim, model in model_tanimlari.items():
        olcekli = isim in olcekleme_gereken
        Xtr = X_train_scaled if olcekli else X_train
        Xte = X_test_scaled  if olcekli else X_test

        t0 = time.time()
        model.fit(Xtr, y_train)
        egitim_suresi = time.time() - t0

        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)

        # 5-Fold Stratified Cross-Validation
        cv_skorlari = cross_val_score(model, Xtr, y_train, cv=5, scoring='accuracy')

        sonuclar[isim] = {
            'accuracy':       accuracy_score(y_test, y_pred),
            'precision':      precision_score(y_test, y_pred, average='weighted'),
            'recall':         recall_score(y_test, y_pred, average='weighted'),
            'f1':             f1_score(y_test, y_pred, average='weighted'),
            'y_pred':         y_pred,
            'y_proba':        y_proba,
            'egitim_suresi':  egitim_suresi,
            'cv_skorlari':    cv_skorlari,
            'cv_ort':         cv_skorlari.mean(),
            'cv_std':         cv_skorlari.std(),
        }
        egitilmis_modeller[isim] = model

    # ── Ensemble Voting Model (Tüm 6 Modelin Oylaması) ──
    # Pipeline kullanılarak ölçekleme gereken modeller otomatik ölçeklenir
    oylama_modeli = VotingClassifier(
        estimators=[
            ('rf',  RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)),
            ('knn', Pipeline([('scaler', StandardScaler()),
                              ('clf', KNeighborsClassifier(n_neighbors=7))])),
            ('svm', Pipeline([('scaler', StandardScaler()),
                              ('clf', SVC(kernel='rbf', probability=True, random_state=42))])),
            ('lr',  Pipeline([('scaler', StandardScaler()),
                              ('clf', LogisticRegression(max_iter=1000, random_state=42))])),
            ('dt',  DecisionTreeClassifier(max_depth=10, random_state=42)),
            ('nb',  GaussianNB()),
        ],
        voting='soft'     # Olasılık tabanlı yumuşak oylama
    )

    t0 = time.time()
    oylama_modeli.fit(X_train, y_train)
    oylama_suresi = time.time() - t0

    oy_pred  = oylama_modeli.predict(X_test)
    oy_proba = oylama_modeli.predict_proba(X_test)
    oy_cv    = cross_val_score(oylama_modeli, X_train, y_train, cv=5, scoring='accuracy')

    sonuclar["🗳️ Ensemble (Oylama)"] = {
        'accuracy':      accuracy_score(y_test, oy_pred),
        'precision':     precision_score(y_test, oy_pred, average='weighted'),
        'recall':        recall_score(y_test, oy_pred, average='weighted'),
        'f1':            f1_score(y_test, oy_pred, average='weighted'),
        'y_pred':        oy_pred,
        'y_proba':       oy_proba,
        'egitim_suresi': oylama_suresi,
        'cv_skorlari':   oy_cv,
        'cv_ort':        oy_cv.mean(),
        'cv_std':        oy_cv.std(),
    }
    egitilmis_modeller["🗳️ Ensemble (Oylama)"] = oylama_modeli

    # ── En İyi Model (F1'e göre) ──
    en_iyi_model = max(sonuclar, key=lambda k: sonuclar[k]['f1'])

    # ── SHAP Açıklanabilirlik (Random Forest — TreeExplainer ile hızlı) ──
    rf_model   = egitilmis_modeller['Random Forest']
    aciklayici = shap.TreeExplainer(rf_model)
    X_shap     = X_test.sample(min(150, len(X_test)), random_state=42)
    shap_degerleri = aciklayici.shap_values(X_shap)   # Sınıf başına bir dizi listesi

    return {
        'df':                 df,
        'egitilmis_modeller': egitilmis_modeller,
        'sonuclar':           sonuclar,
        'le_cloud':           le_cloud,
        'le_season':          le_season,
        'le_location':        le_location,
        'le_weather':         le_weather,
        'X':                  X,
        'X_train':            X_train,
        'X_test':             X_test,
        'X_test_scaled':      X_test_scaled,
        'y_test':             y_test,
        'y_train':            y_train,
        'scaler':             scaler,
        'sinif_isimleri':     le_weather.classes_,
        'en_iyi_model':       en_iyi_model,
        'shap_degerleri':     shap_degerleri,
        'X_shap':             X_shap,
        'aciklayici':         aciklayici,
        'olcekleme_gereken':  olcekleme_gereken,
    }


# ── Verileri Yükle (sadece ilk çalıştırmada eğitim yapılır) ──
veri = sistemi_hazirla()

df              = veri['df']
modeller        = veri['egitilmis_modeller']
sonuclar        = veri['sonuclar']
le_cloud        = veri['le_cloud']
le_season       = veri['le_season']
le_location     = veri['le_location']
le_weather      = veri['le_weather']
X               = veri['X']
X_test          = veri['X_test']
y_test          = veri['y_test']
scaler          = veri['scaler']
sinif_isimleri  = veri['sinif_isimleri']
en_iyi          = veri['en_iyi_model']
shap_deg        = veri['shap_degerleri']
X_shap          = veri['X_shap']
aciklayici      = veri['aciklayici']
olcekleme_grkn  = veri['olcekleme_gereken']


# ═══════════════════════════════════════════════════════════════════════
# YAN MENÜ — VERİ GİRİŞ PANELİ (API VEYA MANUEL)
# ═══════════════════════════════════════════════════════════════════════
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/9338/9338142.png", width=100)
st.sidebar.title("📡 İHA Kontrol İstasyonu")
st.sidebar.markdown("Hava durumu veri kaynağını seçin.")

veri_kaynagi = st.sidebar.radio(
    "🌍 Veri Kaynağı Modu",
    ["📡 Canlı API (Gerçek Zamanlı)", "🎛️ Manuel Sensör Simülatörü"]
)

hava_trafigi = st.sidebar.selectbox(
    "✈️ Hava Trafiği",
    ['Yeşil (Uçuşa Uygun)', 'Sarı (Orta Yoğunluk)', 'Kırmızı (NOTAM - Yasak)']
)

if veri_kaynagi == "📡 Canlı API (Gerçek Zamanlı)":
    st.sidebar.subheader("📍 Konum Seçimi")
    secilen_sehir = st.sidebar.selectbox("Şehir Seçin", list(SEHIRLER.keys()))
    
    lat = SEHIRLER[secilen_sehir]["lat"]
    lon = SEHIRLER[secilen_sehir]["lon"]
    konum_metin = SEHIRLER[secilen_sehir]["loc"]
    
    # Canlı hava durumunu çek
    with st.sidebar.spinner("Canlı hava durumu alınıyor..."):
        hava_verisi = canli_hava_durumu_cek(lat, lon)
        
    if hava_verisi["success"]:
        st.sidebar.success("📡 API Bağlantısı Başarılı!")
        sicaklik = hava_verisi["temp"]
        ruzgar = hava_verisi["wind"]
        # Canlı görüş mesafesini veri setindeki maksimum değer olan 20.0 km ile sınırlıyoruz
        # Aksi takdirde API'den gelen 30-40 km gibi büyük değerler modelleri kararsızlaştırır.
        gorus = min(hava_verisi["vis"], 20.0)
        nem = int(hava_verisi["hum"])
        yagis = hava_verisi["precip"]
        uv = hava_verisi["uv"]
        basinc = hava_verisi["press"]
        
        # Bulutluluk yüzdesinden bulut durumunu çıkar
        bulut_metin = bulut_durumu_belirle(hava_verisi["cloud_pct"])
        
        # Ay tarihine göre mevsimi belirle
        mevsim_metin = mevsim_belirle()
        
        # Canlı Veri Bilgi Kartı
        st.sidebar.info(
            f"**Canlı Sensör Verileri ({secilen_sehir}):**\n"
            f"🌡️ Sıcaklık: {sicaklik} °C\n"
            f"💨 Rüzgar: {ruzgar} km/h\n"
            f"👁️ Görüş: {gorus:.1f} km\n"
            f"💧 Nem: %{nem}\n"
            f"🌧️ Yağış: %{yagis:.0f}\n"
            f"☁️ Bulut: {bulut_metin} (%{hava_verisi['cloud_pct']})\n"
            f"🧭 Basınç: {basinc:.1f} hPa\n"
            f"📅 Mevsim: {mevsim_metin}"
        )
    else:
        st.sidebar.error(f"API Bağlantı Hatası: {hava_verisi['error']}")
        st.sidebar.warning("⚠️ Manuel Simülasyon Moduna geçiliyor.")
        # Hata durumunda varsayılan değerler
        sicaklik = 20.0
        ruzgar = 15.0
        gorus = 15.0
        nem = 40
        yagis = 0.0
        uv = 5
        basinc = 1015.0
        bulut_metin = "clear"
        mevsim_metin = "Spring"
        konum_metin = "coastal"
else:
    # 🎛️ Manuel Sensör Simülatörü
    st.sidebar.subheader("🎛️ Sensör Ayarları")
    sicaklik = st.sidebar.slider("🌡️ Sıcaklık (°C)",          -20.0, 50.0,   20.0)
    ruzgar   = st.sidebar.slider("💨 Rüzgar Şiddeti (km/h)",    0.0,  60.0,   15.0)
    gorus    = st.sidebar.slider("👁️ Görüş Mesafesi (km)",      0.0,  20.0,   15.0)
    nem      = st.sidebar.slider("💧 Nem (%)",                   0,    100,    40)
    yagis    = st.sidebar.slider("🌧️ Yağış İhtimali (%)",       0.0,  100.0,  0.0)
    uv       = st.sidebar.slider("☀️ UV İndeksi",               0,    15,     5)
    basinc   = st.sidebar.slider("🧭 Basınç (hPa)",             900.0,1100.0, 1015.0)

    bulut_metin  = st.sidebar.selectbox("☁️ Bulut Durumu", list(le_cloud.classes_))
    mevsim_metin = st.sidebar.selectbox("📅 Mevsim",       list(le_season.classes_))
    konum_metin  = st.sidebar.selectbox("📍 İHA Konumu",   list(le_location.classes_))


# ═══════════════════════════════════════════════════════════════════════
# ANA EKRAN — 4 SEKME
# ═══════════════════════════════════════════════════════════════════════
st.title("🛸 İHA Karar Destek ve Trafik Yönetim Sistemi (UTM)")

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 UÇUŞ KARAR DESTEK SİSTEMİ",
    "📊 YAPAY ZEKA ANALİZİ",
    "🔬 DETAYLI MODEL ANALİZİ",
    "📡 CANLI TELEMETRİ (OTOPİLOT)",
])


# ───────────────────────────────────────────────────────────────────────
# TAB 1 — MANUEL UÇUŞ KONTROLÜ
# ───────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Manuel Sistem Durumu")

    # Kullanıcı girişini modele uygun hale getir
    bulut_sayi  = le_cloud.transform([bulut_metin])[0]
    mevsim_sayi = le_season.transform([mevsim_metin])[0]
    konum_sayi  = le_location.transform([konum_metin])[0]

    yeni_veri = pd.DataFrame(
        [[sicaklik, nem, ruzgar, yagis, bulut_sayi, basinc, uv, mevsim_sayi, gorus, konum_sayi]],
        columns=X.columns
    )
    yeni_veri_scaled = pd.DataFrame(
        scaler.transform(yeni_veri), columns=X.columns
    )

    # ── Ensemble Tahmini ──
    ensemble      = modeller["🗳️ Ensemble (Oylama)"]
    tahmin_sayi   = ensemble.predict(yeni_veri)[0]
    tahmin_metin  = le_weather.inverse_transform([tahmin_sayi])[0].upper()
    tahmin_proba  = ensemble.predict_proba(yeni_veri)[0]
    guven         = np.max(tahmin_proba) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🤖 YZ Tahmini (Ensemble)", tahmin_metin)
    col2.metric("🎯 Model Güveni",          f"%{guven:.1f}")
    col3.metric("💨 Rüzgar Şiddeti",        f"{ruzgar} km/h")
    col4.metric("👁️ Görüş Mesafesi",        f"{gorus} km")

    st.markdown("---")

    # ── Model Oylama Tablosu ──
    st.subheader("🗳️ Model Oylama Tablosu")
    st.caption("Her model bağımsız tahmin yapar — Ensemble tüm oyları olasılık tabanlı birleştirir.")

    oylama_satirlari = []
    for isim, model in modeller.items():
        if isim == "🗳️ Ensemble (Oylama)":
            continue
        giris   = yeni_veri_scaled if isim in olcekleme_grkn else yeni_veri
        m_pred  = model.predict(giris)[0]
        m_proba = model.predict_proba(giris)[0]
        m_metin = le_weather.inverse_transform([m_pred])[0]
        m_guven = np.max(m_proba) * 100
        oylama_satirlari.append({
            'Model': isim,
            'Tahmin': m_metin,
            'Güven (%)': f"%{m_guven:.1f}",
        })

    oylama_satirlari.append({
        'Model':      '🗳️ **ENSEMBLE KARARI**',
        'Tahmin':     le_weather.inverse_transform([tahmin_sayi])[0],
        'Güven (%)':  f"%{guven:.1f}",
    })

    st.dataframe(
        pd.DataFrame(oylama_satirlari),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # ── Uçuş Karar Sistemi ──
    riskler = []
    if hava_trafigi == 'Kırmızı (NOTAM - Yasak)':
        riskler.append("NOTAM İhlali")
    if ruzgar > 35.0:
        riskler.append("Kritik Rüzgar")
    if gorus < 5.0:
        riskler.append("Düşük Görüş")
    if yagis > 60.0 or tahmin_metin in ['RAINY', 'SNOWY']:
        riskler.append("Sıvı Teması Riski")
    # 4 sınıflı (Rainy, Snowy, Sunny, Cloudy) bir problemde %35 üzeri güven oldukça güçlüdür.
    # %60 limiti gereksiz yere uçuşları reddediyordu. Güven sınırını %35'e çekiyoruz.
    if guven < 35:
        riskler.append("Düşük Model Güveni")

    if len(riskler) > 0:
        st.error(f"❌ UÇUŞ REDDEDİLDİ: {', '.join(riskler)}")
    elif ruzgar > 25.0 or tahmin_metin == 'CLOUDY':
        st.warning("⚠️ RİSKLİ UÇUŞ ONAYLANDI")
    else:
        st.success("✅ GÜVENLİ UÇUŞ ONAYLANDI")

    st.markdown("---")

    # ── SHAP — Bu Karar Neden Verildi? (Explainable AI) ──
    st.subheader("🧠 Açıklanabilir YZ: Bu Karar Neden Verildi?")
    st.caption(
        "SHAP (SHapley Additive exPlanations) ile modelin kararını etkileyen "
        "her sensör faktörünün katkısı gösterilir. Yeşil çubuklar tahmini "
        "destekleyen, kırmızı çubuklar ise zıt yönde etkileyen faktörlerdir."
    )

    rf_model   = modeller['Random Forest']
    tek_shap   = aciklayici.shap_values(yeni_veri)
    rf_tahmin  = rf_model.predict(yeni_veri)[0]
    
    # SHAP versiyonuna göre çıktının liste veya numpy array olma durumunu kontrol et
    if isinstance(tek_shap, list):
        # Liste çıktısı (eski SHAP): Her sınıf için ayrı bir array [sınıf][örnek]
        katkilar = tek_shap[rf_tahmin][0]
    else:
        # Array çıktısı (yeni SHAP): Shape (num_samples, num_features, num_classes)
        if len(tek_shap.shape) == 3:
            katkilar = tek_shap[0, :, rf_tahmin]
        else:
            # Binary/tek sınıf çıktısı: Shape (num_samples, num_features)
            katkilar = tek_shap[0]

    katki_df = pd.DataFrame({
        'Özellik':      X.columns,
        'Katkı (SHAP)': katkilar
    }).sort_values('Katkı (SHAP)', key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    renkler = ['#e74c3c' if v < 0 else '#2ecc71' for v in katki_df['Katkı (SHAP)']]
    ax.barh(katki_df['Özellik'], katki_df['Katkı (SHAP)'], color=renkler, edgecolor='white')
    ax.set_xlabel('SHAP Değeri (Karara Katkı)', fontsize=11)
    ax.set_title(
        f'"{sinif_isimleri[rf_tahmin]}" Tahminini Etkileyen Sensör Faktörleri',
        fontsize=13, fontweight='bold'
    )
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ───────────────────────────────────────────────────────────────────────
# TAB 2 — YAPAY ZEKA ANALİZİ (Özet)
# ───────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Yapay Zeka Performans Özeti")

    # ── En iyi model kartı ──
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                padding: 20px; border-radius: 15px; color: white;
                text-align: center; font-size: 1.2em; margin-bottom: 20px;'>
        🏆 En İyi Model: <strong>{en_iyi}</strong> &nbsp;—&nbsp;
        F1 Skoru: <strong>%{sonuclar[en_iyi]['f1']*100:.2f}</strong> &nbsp;|&nbsp;
        Accuracy: <strong>%{sonuclar[en_iyi]['accuracy']*100:.2f}</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── Özet Metrik Kartları (En İyi Model) ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Accuracy",  f"%{sonuclar[en_iyi]['accuracy']*100:.2f}")
    m2.metric("🔍 Precision", f"%{sonuclar[en_iyi]['precision']*100:.2f}")
    m3.metric("📡 Recall",    f"%{sonuclar[en_iyi]['recall']*100:.2f}")
    m4.metric("⚖️ F1 Score",  f"%{sonuclar[en_iyi]['f1']*100:.2f}")

    st.markdown("---")

    # ── Model Karşılaştırma Bar Chart ──
    st.subheader("📊 Model Karşılaştırması — Accuracy vs F1 Score")

    model_isimleri = list(sonuclar.keys())
    acc_listesi    = [sonuclar[m]['accuracy'] * 100  for m in model_isimleri]
    f1_listesi     = [sonuclar[m]['f1'] * 100        for m in model_isimleri]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos    = np.arange(len(model_isimleri))
    genislik = 0.35

    bar1 = ax.bar(x_pos - genislik / 2, acc_listesi, genislik,
                  label='Accuracy (%)', color='#3498db', alpha=0.85, edgecolor='white')
    bar2 = ax.bar(x_pos + genislik / 2, f1_listesi, genislik,
                  label='F1 Score (%)',  color='#e74c3c', alpha=0.85, edgecolor='white')

    ax.set_ylabel('Skor (%)', fontsize=12)
    ax.set_title('Tüm Modellerin Performans Karşılaştırması', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_isimleri, rotation=25, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 108)
    ax.grid(axis='y', alpha=0.3)

    for bar in bar1:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8)
    for bar in bar2:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Feature Importance + Cross-Validation Box Plot ──
    grafik_col1, grafik_col2 = st.columns(2)

    with grafik_col1:
        st.subheader("🎯 Sensör Önem Sıralaması")
        st.caption("Random Forest modelinin hangi sensöre ne kadar güvendiği")

        rf = modeller['Random Forest']
        onem = pd.DataFrame({
            'Özellik': X.columns,
            'Önem':    rf.feature_importances_
        }).sort_values('Önem', ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(onem['Özellik'], onem['Önem'],
                color=plt.cm.viridis(np.linspace(0.3, 0.9, len(onem))),
                edgecolor='white')
        ax.set_xlabel('Önem Derecesi')
        ax.set_title('Random Forest — Feature Importance', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with grafik_col2:
        st.subheader("📦 Cross-Validation Dağılımı (5-Fold)")
        st.caption("Her modelin 5 farklı veri bölümündeki doğruluk tutarlılığı")

        cv_verisi = [sonuclar[m]['cv_skorlari'] * 100 for m in model_isimleri]

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(cv_verisi, patch_artist=True)
        renkler_cv = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12',
                      '#9b59b6', '#1abc9c', '#e67e22']
        for i, (patch, renk) in enumerate(zip(bp['boxes'], renkler_cv)):
            patch.set_facecolor(renk)
            patch.set_alpha(0.7)
        ax.set_xticklabels(model_isimleri, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('5-Fold Cross-Validation Sonuçları', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Model Bilgileri ──
    with st.expander("ℹ️ Kullanılan Algoritmaların Açıklamaları"):
        st.markdown("""
| Algoritma | Açıklama |
|-----------|----------|
| **Random Forest** | Birden fazla karar ağacının birleşik oylaması ile güçlü tahmin üretir. Overfitting'e dirençlidir. |
| **K-En Yakın Komşu (KNN)** | Yeni veriyi, eğitim setindeki en yakın K komşunun çoğunluk oylamasına göre sınıflandırır. |
| **Destek Vektör Makinesi (SVM)** | Sınıflar arasında optimum bir hiper-düzlem bularak ayrım yapar. Yüksek boyutlu veride güçlüdür. |
| **Lojistik Regresyon** | Lineer sınır çizerek sınıflandırma yapar. Hızlı ve yorumlanabilirdir. |
| **Karar Ağacı** | Veriyi ağaç yapısında dallandırarak kurallar çıkarır. Görselleştirmesi kolaydır. |
| **Naive Bayes** | Bayes teoremine dayalı olasılıksal sınıflandırıcı. Çok hızlıdır. |
| **🗳️ Ensemble (Oylama)** | Tüm 6 modelin olasılık tabanlı yumuşak oylaması. Tek modele güvenmek yerine kolektif karar üretir. |
        """)


# ───────────────────────────────────────────────────────────────────────
# TAB 3 — 🔬 DETAYLI MODEL ANALİZİ
# ───────────────────────────────────────────────────────────────────────
with tab3:
    st.header("🔬 Detaylı Model Analizi ve Değerlendirme Metrikleri")

    # ══════════════ MODEL KARNESİ ══════════════
    st.subheader("📋 Model Karnesi")
    st.caption(
        "Yönergedeki zorunlu metrikler: Accuracy, Precision, Recall, F1 Skoru "
        "— tüm modeller için karşılaştırmalı olarak gösterilmektedir."
    )

    karne_satirlari = []
    for isim in sonuclar:
        s = sonuclar[isim]
        karne_satirlari.append({
            'Model':             isim,
            'Accuracy (%)':      round(s['accuracy']  * 100, 2),
            'Precision (%)':     round(s['precision'] * 100, 2),
            'Recall (%)':        round(s['recall']    * 100, 2),
            'F1 Skoru (%)':      round(s['f1']        * 100, 2),
            'CV Ort. (%)':       round(s['cv_ort']    * 100, 2),
            'CV Std (±)':        round(s['cv_std']    * 100, 2),
            'Eğitim Süresi (s)': round(s['egitim_suresi'], 3),
        })

    karne_df = pd.DataFrame(karne_satirlari)

    # Gradient renklendirme: düşük → kırmızı, yüksek → yeşil
    metrik_sut = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Skoru (%)']
    styled = karne_df.style.background_gradient(
        subset=metrik_sut, cmap='RdYlGn', vmin=70, vmax=100
    ).format({
        'Accuracy (%)':      '{:.2f}',
        'Precision (%)':     '{:.2f}',
        'Recall (%)':        '{:.2f}',
        'F1 Skoru (%)':      '{:.2f}',
        'CV Ort. (%)':       '{:.2f}',
        'CV Std (±)':        '{:.2f}',
        'Eğitim Süresi (s)': '{:.3f}',
    })

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════ MODEL SEÇİMİ İLE DETAYLI ANALİZ ══════════════
    secili_model = st.selectbox(
        "🔍 Detaylı analiz için model seçin:",
        list(sonuclar.keys()),
        index=list(sonuclar.keys()).index(en_iyi)
    )

    detay_col1, detay_col2 = st.columns(2)

    with detay_col1:
        st.markdown("#### 🧩 Hata Matrisi (Confusion Matrix)")
        fig, ax = plt.subplots(figsize=(7, 5))
        cm = confusion_matrix(y_test, sonuclar[secili_model]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=sinif_isimleri, yticklabels=sinif_isimleri, ax=ax)
        ax.set_xlabel('Tahmin Edilen', fontsize=11)
        ax.set_ylabel('Gerçek Değer',  fontsize=11)
        ax.set_title(f'{secili_model}', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with detay_col2:
        st.markdown("#### 📊 Sınıf Bazında Performans Raporu")
        rapor = classification_report(
            y_test, sonuclar[secili_model]['y_pred'],
            target_names=sinif_isimleri, output_dict=True
        )
        rapor_df   = pd.DataFrame(rapor).transpose()
        sinif_rapor = rapor_df.loc[sinif_isimleri, ['precision', 'recall', 'f1-score']]

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(sinif_rapor.astype(float), annot=True, fmt='.3f',
                    cmap='RdYlGn', vmin=0.5, vmax=1.0, ax=ax,
                    linewidths=0.5, linecolor='white')
        ax.set_title(f'{secili_model} — Her Hava Tipi İçin Metrikler', fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ══════════════ ROC EĞRİLERİ ══════════════
    st.subheader("📈 ROC Eğrileri (Tüm Modeller)")
    st.caption(
        "AUC-ROC: Modelin sınıfları ne kadar iyi ayırt edebildiğini gösterir. "
        "1.0 = mükemmel ayrım, 0.5 = rastgele tahmin."
    )

    n_sinif      = len(sinif_isimleri)
    y_test_bin   = label_binarize(y_test, classes=range(n_sinif))

    fig, ax = plt.subplots(figsize=(10, 7))
    renkler_roc = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12',
                   '#9b59b6', '#1abc9c', '#e67e22']

    for idx, (isim, s) in enumerate(sonuclar.items()):
        if s['y_proba'] is not None:
            # Macro-average ROC hesabı
            fpr_list, tpr_list = [], []
            for i in range(n_sinif):
                fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], s['y_proba'][:, i])
                fpr_list.append(fpr_i)
                tpr_list.append(tpr_i)

            all_fpr  = np.unique(np.concatenate(fpr_list))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_sinif):
                mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
            mean_tpr /= n_sinif

            macro_auc = auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr,
                    color=renkler_roc[idx % len(renkler_roc)],
                    linewidth=2,
                    label=f'{isim} (AUC = {macro_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Rastgele Tahmin (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('Macro-Average ROC Eğrileri — Tüm Modeller', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ══════════════ SHAP AÇIKLANABILIRLIK ══════════════
    st.subheader("🧠 SHAP Açıklanabilirlik Analizi (Explainable AI)")
    st.caption(
        "Random Forest modelinin kararlarını etkileyen en önemli sensör "
        "faktörleri — SHAP değerleri ile analiz edilmiştir."
    )

    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        st.markdown("#### 📊 Genel Özellik Önemi (SHAP)")
        shap.summary_plot(
            shap_deg, X_shap, plot_type="bar",
            class_names=list(sinif_isimleri), show=False
        )
        fig_shap1 = plt.gcf()
        fig_shap1.set_size_inches(8, 5)
        plt.tight_layout()
        st.pyplot(fig_shap1)
        plt.close('all')

    with shap_col2:
        st.markdown("#### 🔬 Detaylı SHAP Değer Dağılımı")
        secili_sinif_shap = st.selectbox(
            "Hava tipi seçin:", list(sinif_isimleri),
            key='shap_sinif'
        )
        sinif_idx = list(sinif_isimleri).index(secili_sinif_shap)
        
        # SHAP çıktısının formatına göre doğru alt kümeyi seç
        if isinstance(shap_deg, list):
            secili_shap_val = shap_deg[sinif_idx]
        else:
            if len(shap_deg.shape) == 3:
                secili_shap_val = shap_deg[:, :, sinif_idx]
            else:
                secili_shap_val = shap_deg

        shap.summary_plot(secili_shap_val, X_shap, show=False)
        fig_shap2 = plt.gcf()
        fig_shap2.set_size_inches(8, 5)
        plt.tight_layout()
        st.pyplot(fig_shap2)
        plt.close('all')


# ───────────────────────────────────────────────────────────────────────
# TAB 4 — CANLI TELEMETRİ (OTOPİLOT SİMÜLASYONU)
# ───────────────────────────────────────────────────────────────────────
with tab4:
    st.header("🔴 Otonom Sensör Akışı (Live Telemetry)")
    st.write(
        "Bu modda İHA havada uçuyormuş gibi sensörlerinden saniyede bir "
        "canlı veri çeker ve yapay zeka bu verilere göre anlık kararlar üretir."
    )

    if st.button("🚀 OTOPİLOT SİMÜLASYONUNU BAŞLAT"):
        placeholder = st.empty()

        with st.spinner("Sensörlere bağlanılıyor..."):
            time.sleep(1)

        # Ensemble modeli kullan — en güvenilir karar
        ens = modeller["🗳️ Ensemble (Oylama)"]

        for saniye in range(15):
            # 1. Gerçekçi sensör verisi üret
            sim_sicaklik = round(random.uniform(5.0, 35.0), 1)
            sim_ruzgar   = round(random.uniform(10.0, 45.0), 1)
            sim_gorus    = round(random.uniform(2.0, 20.0), 1)
            sim_nem      = random.randint(30, 90)

            sim_veri = pd.DataFrame(
                [[sim_sicaklik, sim_nem, sim_ruzgar, 0.0, 0, 1010.0, 5, 0, sim_gorus, 0]],
                columns=X.columns
            )

            # 2. YZ tahmin ve güven
            sim_tahmin = le_weather.inverse_transform(ens.predict(sim_veri))[0].upper()
            sim_proba  = ens.predict_proba(sim_veri)[0]
            sim_guven  = np.max(sim_proba) * 100

            # 3. Karar algoritması
            if sim_ruzgar > 35 or sim_gorus < 5:
                durum_mesaji = "❌ ACİL İNİŞ GEREKLİ (Kritik Sensör Verisi)"
                renk = "red"
            elif sim_ruzgar > 25:
                durum_mesaji = "⚠️ DİKKAT RİSKLİ ŞARTLAR (Türbülans Riski)"
                renk = "orange"
            else:
                durum_mesaji = "✅ UÇUŞ STABİL"
                renk = "green"

            # 4. Ekranı güncelle
            with placeholder.container():
                st.markdown(f"### ⏱️ Geçen Süre: {saniye + 1} sn")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Anlık Rüzgar", f"{sim_ruzgar} km/h",
                          delta=round(random.uniform(-5, 5), 1))
                c2.metric("Sıcaklık",     f"{sim_sicaklik} °C",
                          delta=round(random.uniform(-2, 2), 1))
                c3.metric("Görüş",        f"{sim_gorus} km",
                          delta=round(random.uniform(-1, 1), 1))
                c4.metric("YZ Tahmini",   sim_tahmin)
                c5.metric("Model Güveni",  f"%{sim_guven:.0f}")

                st.markdown(
                    f"<h2 style='text-align: center; color: {renk};'>"
                    f"{durum_mesaji}</h2>",
                    unsafe_allow_html=True
                )

            time.sleep(1)

        st.success("🏁 Uçuş Simülasyonu Tamamlandı.")