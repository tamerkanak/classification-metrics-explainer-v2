# 🎯 ML Metrics Akademisi

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Türkçe Machine Learning Classification Metrics'lerini Eğlenceli ve İnteraktif Bir Şekilde Öğrenin!**

Bu kapsamlı Streamlit uygulaması, machine learning classification metriklerini basit ve anlaşılır bir şekilde öğrenmenizi sağlar. Gerçek hayat örnekleri, görsel açıklamalar ve interaktif demonstrasyonlarla donatılmıştır.

## 📊 Demo & Screenshots

Uygulama tamamen Türkçe olup, karmaşık ML kavramlarını basit analogilerle açıklar:

- 🎮 **İnteraktif hesaplayıcılar** - Kendi değerlerinizi girerek sonuçları görebilirsiniz
- 📈 **Plotly grafikleri** - Modern ve etkileşimli görselleştirmeler  
- 🎯 **Confusion matrix'ler** - Renkli ve anlaşılır matrisler
- 💡 **Gerçek hayat örnekleri** - Medical diagnosis, hava durumu tahminleri vb.

## 📚 Kapsanan Metrikler

### Accuracy Metrikleri
1. **⚖️ Balanced Accuracy** - İmbalanced data için adil accuracy ölçümü
2. **🔗 Matthews Correlation Coefficient (MCC)** - Güvenilir korelasyon ölçümü (-1 ile +1 arası)

### Agreement Metrikleri  
3. **🤝 Cohen's Kappa** - İki değerlendirmeci arasındaki uyuşma seviyesi
4. **⚖️ Quadratic Weighted Kappa** - Ordinal data için gelişmiş kappa (hata büyüklüğü önemli)

### Loss Metrikleri
5. **📉 Log Loss** - Probability confidence ve model güven seviyesi ölçümü
6. **🔍 Focal Loss** - Zor örneklere odaklanan akıllı loss (imbalanced data için)

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.8 veya üzeri
- pip package manager

### 1. Repository'yi Klonlayın
```bash
git clone https://github.com/[username]/ml-metrics-akademisi.git
cd ml-metrics-akademisi
```

### 2. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatın
```bash
streamlit run app.py
```

### 4. Tarayıcınızda Açın
Uygulama otomatik olarak `http://localhost:8501` adresinde açılacaktır.

## 🎮 Özellikler ve Yetenekler

### 🔬 Eğitim Odaklı İçerik
- **Adım adım formül açıklamaları** - Her formülün mantığı detaylı anlatılır
- **Tarihçe bilgileri** - Metriklerin kim tarafından, ne zaman ve neden geliştirildiği
- **Gerçek hayat senaryoları** - Medical diagnosis, hava durumu, kanser tespiti örnekleri

### 🎯 İnteraktif Deneyim
- **Canlı hesaplayıcılar** - Confusion matrix değerlerini değiştirerek sonuçları görün
- **Parametre oyun alanları** - Alpha, gamma gibi parametrelerin etkilerini keşfedin
- **Karşılaştırmalı analizler** - Farklı metriklerin aynı veri üzerindeki sonuçları

### 📊 Görsel Zenginlik
- **Modern Plotly grafikleri** - Zoom, pan, hover özellikleri
- **Interaktif confusion matrix'ler** - Tıklanabilir ve bilgi verici
- **Renk kodlu açıklamalar** - Önemli bilgiler vurgulanır
- **Emoji'li tasarım** - Eğlenceli ve akılda kalıcı öğrenme

### 🔍 Detaylı Analizler
- **Imbalanced dataset senaryoları** - Kanser tespiti, fraud detection örnekleri
- **Weight matrix görselleştirmeleri** - Weighted Kappa'da ağırlık etkilerinin görünümü
- **Loss dağılım grafikleri** - Focal Loss'ta kolay vs zor örnek analizleri

## 📖 Kullanım Rehberi

### Başlangıç Seviyesi Kullanıcılar İçin
1. **Ana Sayfadan başlayın** - Genel bir bakış alın
2. **Balanced Accuracy ile devam edin** - En basit metrikten başlayın
3. **İnteraktif bölümleri mutlaka deneyin** - Slider'ları hareket ettirin
4. **"Fun Fact" kutularını okuyun** - Ek bilgiler edinin

### İleri Seviye Kullanıcılar İçin
1. **Focal Loss sayfasını inceleyin** - Imbalanced data stratejileri
2. **Weighted Kappa'yı keşfedin** - Ordinal data için gelişmiş teknikler
3. **Parametre etkilerini analiz edin** - Alpha, gamma optimizasyonu
4. **Gerçek veri örneklerini karşılaştırın** - Pratik uygulamalar

## 🛠 Teknik Detaylar

### Kullanılan Teknolojiler
- **Streamlit** - Web uygulaması framework'ü
- **Plotly** - İnteraktif görselleştirmeler
- **Scikit-learn** - ML metrikleri ve örnek veri
- **NumPy & Pandas** - Veri işleme
- **Matplotlib & Seaborn** - Ek görselleştirmeler

### Proje Yapısı
```
ml-metrics-akademisi/
├── app.py              # Ana Streamlit uygulaması (1560+ satır)
├── Cohen.png           # Cohen's Kappa açıklama görseliKappa calculation visual
├── requirements.txt    # Python bağımlılıkları
├── README.md          # Proje dokumentasyonu
└── __pycache__/       # Python cache dosyaları
```

### Performans ve Optimizasyon
- Efficient caching with Streamlit's caching mechanisms
- Optimized numpy operations for large datasets
- Responsive design for different screen sizes
- Fast loading with minimal dependencies

## 🎯 Hedef Kitle

### Birincil Hedef Kitle
- **ML öğrencileri** ve **yeni başlayanlar**
- **Veri bilimciler** metrics derinleştirmek isteyenler
- **Akademisyenler** öğretim materyali arayanlar
- **Türkçe kaynak** tercih eden geliştiriciler

### İkincil Hedef Kitle
- **Medical AI geliştiricileri** - Diagnostic accuracy için
- **Kaggle yarışmacıları** - Competition metrics için
- **Quality assurance uzmanları** - Model evaluation için

## 🤝 Katkıda Bulunma

Bu projeye katkıda bulunmak istiyorsanız:

1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

### Katkı Alanları
- 🌍 **Dil desteği** - İngilizce, Arapça vb. versiyonlar
- 📊 **Yeni metrikler** - F1-Score, ROC-AUC, Precision-Recall
- 🎨 **UI/UX iyileştirmeleri** - Daha modern tasarım
- 📱 **Mobile optimization** - Mobil cihaz uyumluluğu
- 🔬 **Advanced examples** - Daha karmaşık real-world scenarios

## 📜 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🌟 Teşekkürler

Bu proje aşağıdaki kaynaklardan ilham almıştır:
- **DATAtab** - Weighted Cohen's Kappa tutorial
- **Brian W. Matthews** - MCC metric original paper  
- **Tsung-Yi Lin et al.** - Focal Loss RetinaNet paper
- **Jacob Cohen** - Original Kappa statistic
- **Streamlit Community** - Amazing framework and examples

## 📞 İletişim ve Destek

- 🐛 **Bug reports**: GitHub Issues kullanın
- 💡 **Feature requests**: GitHub Discussions'da tartışın  
- 📧 **Diğer sorular**: README'ye veya kod yorumlarına bakın

---

**Keyifli öğrenmeler! 🎓✨**

> *"Karmaşık kavramları basit örneklerle açıklamak gerçek ustalıktır."* - Bu uygulamanın felsefesi 