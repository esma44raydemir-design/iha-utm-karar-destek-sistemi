# 🛸 AI-Powered UAV Decision Support & Traffic Management System (UTM)
### (Yapay Zeka Destekli İHA Karar Destek ve Trafik Yönetim Sistemi)

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Bu proje, Streamlit ile geliştirilmiş **akademik ve endüstriyel standartlarda bir İHA (Drone) Yer Kontrol ve Uçuş Karar Destek İstasyonudur**. Çoklu Makine Öğrenmesi algoritmalarını, oylama tabanlı bir kolektif karar mekanizmasını, canlı meteorolojik API entegrasyonunu ve Açıklanabilir Yapay Zekayı (XAI) bir araya getirerek gerçek zamanlı ve yüksek güvenilirlikli otonom uçuş izni kararları üretir.

This project is an advanced **UAV (Drone) Ground Control and Flight Decision Support Station** built with Streamlit. It integrates multiple Machine Learning algorithms, an ensemble voting mechanism, live meteorological API data, and Explainable AI (XAI) to produce high-reliability, real-time autonomous flight clearance decisions.

---

## 🚀 Öne Çıkan Özellikler (Key Features)

### 1. Çoklu Model Makine Öğrenmesi (6 ML Algoritması)
Sistem, geçmiş hava sınıflandırma verilerini kullanarak 6 farklı makine öğrenmesi algoritmasını eğitir ve karşılaştırır:
* **Random Forest** (Rastgele Orman)
* **K-Nearest Neighbors** (K-En Yakın Komşu - KNN)
* **Support Vector Machines** (Destek Vektör Makineleri - SVM)
* **Logistic Regression** (Lojistik Regresyon)
* **Decision Trees** (Karar Ağaçları)
* **Naive Bayes** (Olasılıksal Sınıflandırıcı)

### 2. Yüksek Güvenilirlikli Kolektif Karar Mekanizması (Voting Ensemble)
İHA uçuş kararlarının güvenliği için tek bir yapay zeka modeline güvenmek yerine, 6 modelin olasılık tabanlı ortak oylaması (**Soft Voting Ensemble**) kullanılmıştır. Bu yaklaşım, sistemin tek hata noktası riskini (Single Point of Failure) ortadan kaldırır.

### 3. Açıklanabilir Yapay Zeka (SHAP - Explainable AI)
**SHAP (SHapley Additive exPlanations)** kütüphanesi entegrasyonu sayesinde sistem yapay zekayı bir "kara kutu" olmaktan çıkarır. Alınan kararların (onay veya red) hangi sensör verilerinden (rüzgar, nem, görüş vb.) ne ölçüde etkilendiği, pilotlar ve yer kontrolörleri için grafiklerle şeffaf bir şekilde açıklanır.

### 4. Canlı Hava Durumu API Entegrasyonu (Open-Meteo)
* **Sıfır Manuel Girdi**: Türkiye'nin 8 büyük ili (İstanbul, Ankara, İzmir, Erzurum, Antalya, Trabzon, Eskişehir, Malatya) için koordinat tabanlı entegrasyon sağlanmıştır.
* **Open-Meteo API**: Seçilen şehre ait anlık sıcaklık, nem, rüzgar hızı, görüş mesafesi, basınç ve UV verileri internetten otomatik çekilir.
* **Akıllı Eşleme**: API'den gelen canlı veriler (bulutluluk yüzdesi, tarih vb.) otomatik olarak makine öğrenmesi modellerinin eğitim formatına dönüştürülür.

### 5. Canlı Telemetri Otopilot Simülatörü
İHA havada otonom olarak uçuyormuş gibi saniyede bir değişen canlı sensör verileri simüle edilir. Kritik sensör sınırları aşıldığında otopilot otonom olarak acil iniş kararı üretir.

---

## 📊 Kapsamlı Model Değerlendirme Karnesi
Projenin "Detaylı Model Analizi" sekmesinde yer alan akademik değerlendirme araçları:
* **Model Karnesi**: Tüm modellerin **Accuracy** (Doğruluk), **Precision** (Kesinlik), **Recall** (Duyarlılık), **F1-Skoru** ve eğitim sürelerinin karşılaştırmalı tablosu.
* **5-Fold Cross-Validation (Çapraz Doğrulama)**: Modellerin ezberlemediğini (overfitting olmadığını) kanıtlayan Box Plot dağılım analizleri.
* **Dinamik Hata Matrisi (Confusion Matrix)**: Seçilen modelin sınıflandırma hatalarını gösteren heatmap grafiği.
* **Macro-Average ROC Eğrisi**: Modellerin sınıfları ayırt etme yeteneğini ve AUC (Alan Altındaki Alan) değerlerini karşılaştıran grafik.

---

## ⚙️ Kurulum ve Çalıştırma (Installation & Usage)

### Gereksinimler
Sistemde Python 3.9 veya daha yeni bir sürümün yüklü olduğundan emin olun.

1. **Projeyi Klonlayın:**
   ```bash
   git clone https://github.com/KULLANICI_ADINIZ/uav-ai-decision-support-utm.git
   cd uav-ai-decision-support-utm
