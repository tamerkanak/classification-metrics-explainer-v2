import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    log_loss, confusion_matrix, classification_report
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎯 ML Metrics Akademisi",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .fun-fact {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #ff4757;
    }
    .explanation-box {
        background: #f8f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e1e8ed;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample classification data for demonstrations"""
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                             n_informative=7, n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
    return y_test, y_pred, y_pred_proba

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Create an interactive confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(cm, 
                    labels=dict(x="Tahmin Edilen", y="Gerçek", color="Sayı"),
                    x=[f'Sınıf {i}' for i in range(len(cm))],
                    y=[f'Sınıf {i}' for i in range(len(cm))],
                    color_continuous_scale='Blues',
                    title=title)
    
    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            fig.add_annotation(x=j, y=i, text=str(cm[i][j]),
                             showarrow=False, font_color="white" if cm[i][j] > cm.max()/2 else "black")
    
    return fig

def homepage():
    """Main homepage with overview"""
    st.markdown('<h1 class="main-header">🎯 Machine Learning Metrics Akademisi</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
    <h2>🚀 Hoş Geldiniz!</h2>
    <p>Bu eğlenceli akademide machine learning'in en önemli classification metriklerini öğreneceksiniz! 
    Her bir metriği basit örnekler, görsel açıklamalar ve interaktif demonstrasyonlarla keşfedeceğiz.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Accuracy Metrikleri</h3>
        <p>• Balanced Accuracy<br>
        • Matthews Correlation Coefficient</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🤝 Agreement Metrikleri</h3>
        <p>• Cohen's Kappa<br>
        • Quadratic Weighted Kappa</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📉 Loss Metrikleri</h3>
        <p>• Log Loss<br>
        • Focal Loss</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="fun-fact">
    <h3>🎓 Öğrenme İpucu</h3>
    <p>Sol taraftaki menüden istediğiniz metriği seçin ve deep dive yapın! 
    Her sayfa interaktif örnekler ve görsel açıklamalar içeriyor.</p>
    </div>
    """, unsafe_allow_html=True)

def balanced_accuracy_page():
    """Balanced Accuracy explanation page"""
    st.markdown("# ⚖️ Balanced Accuracy")
    
    st.markdown("""
    <div class="explanation-box">
    <h2>🤔 Balanced Accuracy Nedir?</h2>
    <p><strong>Basit açıklama:</strong> Normal accuracy'nin daha adil versiyonu! 
    Özellikle veri setindeki sınıflar eşit sayıda olmadığında (imbalanced data) çok işe yarar.</p>
    
    <p><strong>Gerçek hayat analojisi:</strong> Sınıfta 90 kız 10 erkek öğrenci var. 
    "Herkesi kız" diye tahmin etsen doğru olur ama bu adil mi? 
    Balanced Accuracy her sınıfın başarısını eşit ağırlıkta değerlendirir!</p>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    \text{Naive Accuracy} = \frac{90}{100} = 90\% \text{ ama erkekleri hiç tespit etmiyor!}
    ''')
    
    st.markdown("""
    </div>
    """, unsafe_allow_html=True)
    
    # Formula
    st.markdown("### 📐 Formül")
    
    st.latex(r'''
    \text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}
    ''')
    
    st.latex(r'''
    = \frac{\frac{TP}{TP + FN} + \frac{TN}{TN + FP}}{2}
    ''')
    
    st.markdown("**Alt formüller:**")
    
    st.latex(r'''
    \text{Sensitivity (Recall)} = \frac{TP}{TP + FN}
    ''')
    
    st.latex(r'''
    \text{Specificity} = \frac{TN}{TN + FP}
    ''')
    
    st.markdown("""
    **🧠 Formülün Mantığı:**
    
    Bu formül **her sınıfın performansını eşit ağırlıkta** değerlendirmek için tasarlandı:
    
    - **Sensitivity (Recall)** → Pozitif sınıfın ne kadarını doğru yakaladık?
    - **Specificity** → Negatif sınıfın ne kadarını doğru yakaladık?
    - **Ortalama** → Her iki sınıfa da eşit saygı!
    """)
    
    st.latex(r'''
    \text{Sensitivity} = \frac{TP}{TP + FN} \quad \text{Specificity} = \frac{TN}{TN + FP}
    ''')
    
    st.markdown("""
    **💡 Neden Doğdu Bu Metrik?**
    Normal accuracy imbalanced data'da yanıltıcı olur. 100 hastadan 95'i sağlıklı, 5'i hasta olsun. 
    "Herkesi sağlıklı" desen accuracy alırsın ama hasta olanları hiç tespit etmemişsin! 
    Balanced Accuracy böyle aldatmacalara kanmaz.
    """)
    
    st.latex(r'''
    \text{Normal Accuracy} = \frac{\text{Doğru Tahminler}}{\text{Toplam Tahminler}} = \frac{95}{100} = 0.95
    ''')
    
    st.markdown("""
    **Ama gerçekte hasta tespiti:** """)
    
    st.latex(r'''
    \text{Sensitivity} = \frac{0}{5} = 0 \quad \text{(Hiç hasta tespit edilemedi!)}
    ''')
    
    st.markdown("""
    """)
    
    st.markdown("""
    <div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196F3;">
    <strong>📚 Tarihçe:</strong> Bu metrik özellikle medical diagnosis ve biyoinformatikte yaygınlaştı. 
    Çünkü hasta olan birini kaçırmak (false negative) ile sağlıklı birini yanlış alarm (false positive) 
    vermek arasında denge kurması gerekiyordu.
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive demo
    st.markdown("### 🎮 İnteraktif Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚙️ Senaryoyu Ayarla:**")
        class_0_size = st.slider("Sınıf 0 örneklerinin sayısı", 10, 200, 150)
        class_1_size = st.slider("Sınıf 1 örneklerinin sayısı", 10, 200, 50)
        accuracy_class_0 = st.slider("Sınıf 0 doğru tahmin oranı (%)", 0, 100, 80) / 100
        accuracy_class_1 = st.slider("Sınıf 1 doğru tahmin oranı (%)", 0, 100, 70) / 100
    
    with col2:
        # Calculate metrics
        correct_0 = int(class_0_size * accuracy_class_0)
        correct_1 = int(class_1_size * accuracy_class_1)
        total_correct = correct_0 + correct_1
        total_samples = class_0_size + class_1_size
        
        normal_accuracy = total_correct / total_samples
        balanced_acc = (accuracy_class_0 + accuracy_class_1) / 2
        
        st.markdown("**📊 Sonuçlar:**")
        st.metric("Normal Accuracy", f"{normal_accuracy:.2%}")
        st.metric("Balanced Accuracy", f"{balanced_acc:.2%}")
        
        # Visual comparison
        fig = go.Figure(data=[
            go.Bar(name='Normal Accuracy', x=['Metrik'], y=[normal_accuracy], marker_color='lightblue'),
            go.Bar(name='Balanced Accuracy', x=['Metrik'], y=[balanced_acc], marker_color='lightcoral')
        ])
        fig.update_layout(title='Accuracy Karşılaştırması', yaxis_title='Değer')
        st.plotly_chart(fig, use_container_width=True)
    
    # Real example
    st.markdown("### 🔬 Gerçek Veri Örneği")
    y_true, y_pred, _ = create_sample_data()
    
    normal_acc = np.mean(y_true == y_pred)
    balanced_acc_real = balanced_accuracy_score(y_true, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal Accuracy", f"{normal_acc:.3f}")
    with col2:
        st.metric("Balanced Accuracy", f"{balanced_acc_real:.3f}")
    
    # Confusion matrix
    fig = plot_confusion_matrix(y_true, y_pred, "Örnek Veri Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="fun-fact">
    <h3>💡 Ne Zaman Kullanmalı?</h3>
    <p>• Veri setinde sınıflar eşit dağılmadığında<br>
    • Her sınıfın eşit önemde olduğu durumlarda<br>
    • Medical diagnosis gibi kritik uygulamalarda</p>
    </div>
    """, unsafe_allow_html=True)

def matthews_correlation_page():
    """Matthews Correlation Coefficient explanation page"""
    st.markdown("# 🔗 Matthews Correlation Coefficient (MCC)")
    
    st.markdown("""
    <div class="explanation-box">
    <h2>🤔 MCC Nedir?</h2>
    <p><strong>Basit açıklama:</strong> Machine learning dünyasının korelasyon katsayısı! 
    -1 ile +1 arasında değer alır ve modelin ne kadar güvenilir olduğunu söyler.</p>
    
    <p><strong>Gerçek hayat analojisi:</strong> Hava durumu tahmininde:<br>
    • +1: Mükemmel tahmin (her zaman doğru)<br>
    • 0: Rastgele tahmin kadar iyi<br>
    • -1: Her zaman yanlış (bu da aslında bir bilgi!)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Formula explanation
    st.markdown("### 📐 Formül")
    
    st.latex(r'''
    MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
    ''')
    
    st.markdown("""
    **🧠 Formülün Mantığı:**
    
    Bu formül **Pearson korelasyon katsayısının** binary classification'a uyarlanmış hali:
    
    - **Pay (Numerator)**: "İyi tahminler" - "Kötü tahminler"
    - **Payda (Denominator)**: Normalizasyon faktörü → Sonucun -1 ile +1 arasında olmasını sağlar
    """)
    
    st.latex(r'''
    \text{Pay} = TP \times TN - FP \times FN
    ''')
    
    st.latex(r'''
    \text{Payda} = \sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}
    ''')
    
    st.markdown("""
    
    **💡 Neden Bu Formül?**
    """)
    
    st.latex(r'''
    TP \times TN: \text{Hem pozitif hem negatif sınıfı doğru tahmin (iyi!)}
    ''')
    
    st.latex(r'''
    FP \times FN: \text{Pozitifi negatif, negatifi pozitif tahmin (kötü!)}
    ''')
    
    st.latex(r'''
    \text{Fark} = (TP \times TN) - (FP \times FN): \text{Net başarımız}
    ''')
    
    st.markdown("""
    - **Karekök**: Tüm confusion matrix değerlerinin etkisini dengeler
    """)
    
    st.markdown("""
    <div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #4169E1;">
    <strong>📚 Tarihçe:</strong> Brian W. Matthews tarafından 1975'te protein structure prediction için geliştirildi. 
    O zamanlar da imbalanced biological data'yla uğraşıyorlardı! Formül, matematiksel olarak 
    Phi coefficient (φ) ile aynıdır - bu da 2x2 tablo için chi-square testinin özel halidir.
    </div>
    """, unsafe_allow_html=True)
    
    # Visual explanation
    st.markdown("### 📊 MCC Değerlerinin Anlamı")
    
    mcc_values = [-1, -0.5, 0, 0.5, 1]
    mcc_meanings = [
        "Tamamen Yanlış 😱",
        "Kötü Model 😞", 
        "Rastgele Tahmin 🎲",
        "İyi Model 😊",
        "Mükemmel Model 🎯"
    ]
    
    fig = go.Figure()
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    
    for i, (val, meaning, color) in enumerate(zip(mcc_values, mcc_meanings, colors)):
        fig.add_trace(go.Scatter(
            x=[val], y=[i], 
            mode='markers+text',
            marker=dict(size=20, color=color),
            text=f"{val}<br>{meaning}",
            textposition="middle right",
            name=meaning
        ))
    
    fig.update_layout(
        title="MCC Değer Skalası",
        xaxis_title="MCC Değeri",
        yaxis=dict(showticklabels=False),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive calculator
    st.markdown("### 🧮 MCC Hesaplayıcı")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Confusion Matrix Değerlerini Girin:**")
        tp = st.number_input("True Positive (TP)", min_value=0, value=85)
        tn = st.number_input("True Negative (TN)", min_value=0, value=90)
        fp = st.number_input("False Positive (FP)", min_value=0, value=10)
        fn = st.number_input("False Negative (FN)", min_value=0, value=15)
    
    with col2:
        # Calculate MCC
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            mcc = 0
        else:
            mcc = numerator / denominator
        
        st.markdown("**📊 Sonuç:**")
        st.metric("MCC Değeri", f"{mcc:.3f}")
        
        # Interpretation
        if mcc > 0.8:
            interpretation = "🎯 Mükemmel!"
        elif mcc > 0.6:
            interpretation = "😊 Çok İyi!"
        elif mcc > 0.4:
            interpretation = "👍 İyi"
        elif mcc > 0.2:
            interpretation = "😐 Orta"
        elif mcc > 0:
            interpretation = "👎 Zayıf"
        else:
            interpretation = "😱 Kötü"
        
        st.markdown(f"**Değerlendirme:** {interpretation}")
        
        # Visual confusion matrix
        cm_data = [[tp, fp], [fn, tn]]
        fig = px.imshow(cm_data, 
                       labels=dict(x="Tahmin", y="Gerçek"),
                       x=['Positive', 'Negative'],
                       y=['Positive', 'Negative'],
                       title="Confusion Matrix Görselleştirme")
        
        for i in range(2):
            for j in range(2):
                fig.add_annotation(x=j, y=i, text=str(cm_data[i][j]),
                                 showarrow=False, font_color="white")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Real example
    st.markdown("### 🔬 Gerçek Veri Örneği")
    y_true, y_pred, _ = create_sample_data()
    
    # For binary classification, we'll use first two classes
    binary_mask = (y_true <= 1) & (y_pred <= 1)
    y_true_binary = y_true[binary_mask]
    y_pred_binary = y_pred[binary_mask]
    
    if len(y_true_binary) > 0:
        mcc_real = matthews_corrcoef(y_true_binary, y_pred_binary)
        st.metric("Örnek Veri MCC", f"{mcc_real:.3f}")
    
    st.markdown("""
    <div class="fun-fact">
    <h3>🌟 MCC'nin Süper Güçleri</h3>
    <p>• İmbalanced data'da bile güvenilir<br>
    • Tüm confusion matrix değerlerini dikkate alır<br>
    • Karşılaştırma yapmak için mükemmel (-1 to +1 range)<br>
    • Medical diagnosis için altın standart</p>
    </div>
    """, unsafe_allow_html=True)

def cohens_kappa_page():
    """Cohen's Kappa explanation page"""
    st.markdown("# 🤝 Cohen's Kappa")
    
    st.markdown("""
    <div class="explanation-box">
    <h2>🤔 Cohen's Kappa Nedir?</h2>
    <p><strong>Basit açıklama:</strong> İki değerlendirmeci arasındaki uyuşma seviyesini ölçer! 
    Sadece "şans eseri uyuşma"yı çıkarıp gerçek uyuşmayı bulur.</p>
    
    <p><strong>Medical örnek:</strong> Depresyon teşhisi için yeni bir araç geliştirdiniz. 
    İki doktor bu aracı kullanarak 50 hastayı değerlendiriyor. 
    İkisi de aynı sonuca varıyor mu, yoksa şans eseri mi uyuşuyorlar? 
    Cohen's Kappa tam da bunu ölçer!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Formula explanation
    st.markdown("### 📐 Formül")
    
    st.latex(r'''
    \kappa = \frac{P_o - P_e}{1 - P_e}
    ''')
    
    st.markdown("**Açılımı:**")
    
    st.latex(r'''
    P_o = \frac{\text{Gözlenen Uyuşma}}{\text{Toplam}} \quad P_e = \sum_{i} \frac{n_{i+} \times n_{+i}}{N^2}
    ''')
    
    # Add Cohen.png image
    st.image("Cohen.png", caption="Cohen's Kappa Hesaplama Örneği", use_container_width=True)
    
    st.markdown("""
    **🧠 Adım Adım Hesaplama Örneği:**
    
    **🏥 Senaryo:** İki doktor (Rater 1 ve Rater 2) 50 hastayı "Depresif" veya "Depresif Değil" olarak değerlendiriyor.
    
    **📊 Sonuçlar:**
    - Her ikisi de "Depresif Değil" dedi: 17 hasta
    - Her ikisi de "Depresif" dedi: 19 hasta  
    - Rater 1 "Depresif Değil", Rater 2 "Depresif": 8 hasta
    - Rater 1 "Depresif", Rater 2 "Depresif Değil": 6 hasta
    
    **📏 Hesaplama:**
    
    **1. Adım - Gözlenen Uyuşma (Po):**
    """)
    
    st.latex(r'''
    P_o = \frac{17 + 19}{50} = \frac{36}{50} = 0.72 = 72\%
    ''')
    
    st.markdown("""
    **2. Adım - Beklenen Uyuşma (Pe):**
    """)
    
    st.latex(r'''
    \text{Rater 1: } \frac{25}{50} = 50\% \text{ "Depresif Değil", } \frac{25}{50} = 50\% \text{ "Depresif"}
    ''')
    
    st.latex(r'''
    \text{Rater 2: } \frac{23}{50} = 46\% \text{ "Depresif Değil", } \frac{27}{50} = 54\% \text{ "Depresif"}
    ''')
    
    st.latex(r'''
    P_e = (0.50 \times 0.46) + (0.50 \times 0.54) = 0.23 + 0.27 = 0.50
    ''')
    
    st.markdown("""
    **3. Adım - Cohen's Kappa:**
    """)
    
    st.latex(r'''
    \kappa = \frac{P_o - P_e}{1 - P_e} = \frac{0.72 - 0.50}{1 - 0.50} = \frac{0.22}{0.50} = \mathbf{0.44}
    ''')
    
    st.markdown("""
    """)
    
    st.markdown("""
    <div style="background: #f9f7ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #9C27B0;">
    <strong>📚 Tarihçe:</strong> Jacob Cohen tarafından 1960'da psikoloji araştırmaları için geliştirildi. 
    O dönem iki psikolog aynı hastayı değerlendirdiğinde ne kadar uyuştukları merak ediliyordu. 
    Cohen, sadece "kaç tanede uyuştular" değil, "bu uyuşma ne kadar anlamlı" sorusunu sordu.
    </div>
    """, unsafe_allow_html=True)
    
    # Kappa interpretation scale
    st.markdown("### 📊 Kappa Değer Skalası")
    
    kappa_ranges = [
        (-1.0, 0.0, "Kötüden Kötü 😱", "#8B0000", "white"),  # Dark red with white text
        (0.0, 0.20, "Zayıf Uyuşma 😞", "#DC143C", "white"),  # Crimson with white text
        (0.21, 0.40, "Adil Uyuşma 😐", "#FF8C00", "black"),  # Dark orange with black text
        (0.41, 0.60, "Orta Uyuşma 👍", "#FFD700", "black"),  # Gold with black text
        (0.61, 0.80, "İyi Uyuşma 😊", "#32CD32", "black"),  # Lime green with black text
        (0.81, 1.00, "Çok İyi Uyuşma 🎯", "#228B22", "white")  # Forest green with white text
    ]
    
    fig = go.Figure()
    
    for i, (min_val, max_val, meaning, bg_color, text_color) in enumerate(kappa_ranges):
        # Add colored rectangle with border
        fig.add_shape(
            type="rect",
            x0=min_val, y0=i-0.35, x1=max_val, y1=i+0.35,
            fillcolor=bg_color, 
            opacity=0.9, 
            line=dict(color="black", width=2)
        )
        
        # Add text annotation with better contrast
        fig.add_annotation(
            x=(min_val + max_val)/2, y=i,
            text=f"<b>{min_val:.2f} - {max_val:.2f}</b><br><b>{meaning}</b>",
            showarrow=False, 
            font=dict(size=16, color=text_color, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)" if text_color == "black" else "rgba(0,0,0,0.3)",
            bordercolor="black",
            borderwidth=1
        )
    
    fig.update_layout(
        title=dict(
            text="<b>Cohen's Kappa Değerlendirme Skalası (Landis & Koch)</b>",
            font=dict(size=18, color="darkblue"),
            x=0.5
        ),
        xaxis=dict(
            title=dict(
                text="<b>Kappa Değeri</b>",
                font=dict(size=14, color="darkblue")
            ),
            tickfont=dict(size=12),
            gridcolor="lightgray",
            gridwidth=1
        ),
        yaxis=dict(
            showticklabels=False, 
            range=[-0.6, 5.6],
            showgrid=False
        ),
        height=480,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive example
    st.markdown("### 🎮 İnteraktif Kappa Hesaplayıcı")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Senaryo: Medical Değerlendirme**")
        st.markdown("İki doktor hastaları 'Depresif' veya 'Depresif Değil' olarak değerlendiriyor:")
        
        both_not_depressed = st.slider("İkisi de 'Depresif Değil' dediği hastalar", 0, 100, 17)
        both_depressed = st.slider("İkisi de 'Depresif' dediği hastalar", 0, 100, 19)
        doctor1_not_doctor2_depressed = st.slider("1. Depresif Değil, 2. Depresif dediği hastalar", 0, 50, 8)
        doctor1_depressed_doctor2_not = st.slider("1. Depresif, 2. Depresif Değil dediği hastalar", 0, 50, 6)
    
    with col2:
        # Calculate observed and expected agreement
        total = both_not_depressed + both_depressed + doctor1_not_doctor2_depressed + doctor1_depressed_doctor2_not
        
        if total > 0:
            observed_agreement = (both_not_depressed + both_depressed) / total
            
            # Expected agreement by chance
            doctor1_not_total = both_not_depressed + doctor1_not_doctor2_depressed
            doctor1_depressed_total = both_depressed + doctor1_depressed_doctor2_not
            doctor2_not_total = both_not_depressed + doctor1_depressed_doctor2_not
            doctor2_depressed_total = both_depressed + doctor1_not_doctor2_depressed
            
            expected_not_depressed = (doctor1_not_total / total) * (doctor2_not_total / total)
            expected_depressed = (doctor1_depressed_total / total) * (doctor2_depressed_total / total)
            expected_agreement = expected_not_depressed + expected_depressed
            
            if expected_agreement < 1.0:
                kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
            else:
                kappa = 1.0
            
            st.markdown("**📊 Sonuçlar:**")
            st.metric("Gözlenen Uyuşma (Po)", f"{observed_agreement:.3f} ({observed_agreement:.1%})")
            st.metric("Beklenen Uyuşma (Pe)", f"{expected_agreement:.3f} ({expected_agreement:.1%})")
            st.metric("Cohen's Kappa", f"{kappa:.3f}")
            
            # Interpretation
            for min_val, max_val, meaning, bg_color, text_color in kappa_ranges:
                if min_val <= kappa <= max_val:
                    st.markdown(f"**Değerlendirme:** {meaning}")
                    break
        else:
            st.warning("Toplam hasta sayısı 0 olamaz!")
    
    # Confusion matrix visualization
    if total > 0:
        st.markdown("### 📊 Uyuşma Matrisi")
        confusion_data = [[both_not_depressed, doctor1_not_doctor2_depressed], 
                         [doctor1_depressed_doctor2_not, both_depressed]]
        
        fig = px.imshow(confusion_data,
                       labels=dict(x="Doktor 2", y="Doktor 1"),
                       x=['Depresif Değil', 'Depresif'], y=['Depresif Değil', 'Depresif'],
                       title="Doktorların Uyuşma Matrisi")
        
        for i in range(2):
            for j in range(2):
                fig.add_annotation(x=j, y=i, text=str(confusion_data[i][j]),
                                 showarrow=False, font_color="white")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Real example
    st.markdown("### 🔬 Gerçek Veri Örneği")
    y_true, y_pred, _ = create_sample_data()
    kappa_real = cohen_kappa_score(y_true, y_pred)
    
    st.metric("Örnek Veri Kappa", f"{kappa_real:.3f}")
    
    st.markdown("""
    <div class="fun-fact">
    <h3>🎯 Kappa'nın Kullanım Alanları</h3>
    <p>• Medical diagnosis doğruluğu<br>
    • Makine çevirisi kalitesi<br>
    • Annotation quality control<br>
    • Multi-rater studies</p>
    </div>
    """, unsafe_allow_html=True)

def quadratic_weighted_kappa_page():
    """Quadratic Weighted Kappa explanation page"""
    st.markdown("# ⚖️ Quadratic Weighted Kappa")
    
    st.markdown("""
    <div class="explanation-box">
    <h2>🤔 Quadratic Weighted Kappa Nedir?</h2>
    <p><strong>Basit açıklama:</strong> Normal Kappa'nın gelişmiş versiyonu! 
    <strong>Ordinal (sıralı) değişkenler</strong>  için kullanılır. Hataların büyüklüğünü de dikkate alır.</p>
    
    <p><strong>Medical örnek:</strong> İki doktor hasta memnuniyetini değerlendiriyor:<br>
    • <strong>Memnun Değil</strong> → <strong>Nötr</strong> → <strong>Memnun</strong> (sıralı kategoriler)<br>
    • "Memnun Değil" yerine "Nötr" demek → Küçük hata<br>
    • "Memnun Değil" yerine "Memnun" demek → Büyük hata<br>
    Quadratic Kappa büyük hataları çok daha ağır cezalandırır!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add visual explanation of nominal vs ordinal
    st.markdown("### 📊 Nominal vs Ordinal Farkı")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <strong>🔤 Nominal (Normal Kappa):</strong><br>
        - Sıra YOK<br>
        - Kategoriler: Kedi, Köpek, Kuş<br>
        - Tüm hatalar eşit
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <strong>📏 Ordinal (Weighted Kappa):</strong><br>
        - Sıra VAR<br>
        - Kategoriler: Memnun Değil < Nötr < Memnun<br>
        - Hata büyüklüğü önemli
        """, unsafe_allow_html=True)
    
    # Reference to DATAtab
    st.markdown("""
    <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; border-left: 4px solid #4CAF50;">
    <strong>📚 Kaynak:</strong> Bu anlatım <a href="https://datatab.net/tutorial/weighted-cohens-kappa" target="_blank">DATAtab Weighted Cohen's Kappa</a> 
    tutorialından adapte edilmiştir. Ordinal data için weighted kappa kullanımı detaylı şekilde açıklanmıştır.
    </div>
    """, unsafe_allow_html=True)
    
    # Formula explanation
    st.markdown("### 📐 Formül")
    
    st.latex(r'''
    \kappa_w = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}
    ''')
    
    st.markdown("**Ağırlık matrisi (Quadratic):**")
    
    st.latex(r'''
    w_{ij} = \frac{(i-j)^2}{(N-1)^2}
    ''')
    
    st.markdown("### 🔄 Linear vs Quadratic Weighting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<strong>📏 Linear Weighting:</strong>", unsafe_allow_html=True)
        st.latex(r'''
        w_{ij} = \frac{|i-j|}{N-1}
        ''')
        st.markdown("Mesafeyle doğru orantılı ceza")
        
    with col2:
        st.markdown("<strong>📐 Quadratic Weighting:</strong>", unsafe_allow_html=True)
        st.latex(r'''
        w_{ij} = \frac{(i-j)^2}{(N-1)^2}
        ''')
        st.markdown("Mesafenin karesi kadar ceza!")
    
    st.markdown("""
    <strong>🧠 3 Kategorili Örnek (Memnun Değil=0, Nötr=1, Memnun=2):</strong>
    """, unsafe_allow_html=True)
    
    # Create comparison table for 3x3 case
    weighting_comparison = pd.DataFrame({
        'Hata': ['0→1', '0→2', '1→0', '1→2', '2→0', '2→1'],
        'Mesafe': [1, 2, 1, 1, 2, 1],
        'Linear Weight': [0.5, 1.0, 0.5, 0.5, 1.0, 0.5],
        'Quadratic Weight': [0.25, 1.0, 0.25, 0.25, 1.0, 0.25]
    })
    st.dataframe(weighting_comparison, use_container_width=True)
    
    st.markdown("""
    <strong>💡 Neden Quadratic Weight?</strong><br>
    Quadratic weighting büyük hataları çok daha ağır cezalandırır:
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #fff3e0; padding: 1rem; border-radius: 8px; border-left: 4px solid #FF9800;">
    <strong>📚 Tarihçe:</strong> Bu metrik özellikle <strong> ordinal (sıralı) kategoriler</strong>  için geliştirildi. 
    Tıpta hastalık severity (hafif→orta→şiddetli), eğitimde notlar (A→B→C→D→F), 
    psikolojide likert scales gibi durumlarda kullanılır. "5 yerine 4 demek, 5 yerine 1 demekten çok daha az kötü!"
    </div>
    """, unsafe_allow_html=True)
    
    # Weight matrix visualization
    st.markdown("### ⚖️ Ağırlık Matrisi Nasıl Çalışır?")
    
    # Create weight matrix example for 5 classes
    n_classes = 5
    weight_matrix = np.zeros((n_classes, n_classes))
    
    for i in range(n_classes):
        for j in range(n_classes):
            weight_matrix[i, j] = (i - j) ** 2 / (n_classes - 1) ** 2
    
    fig = px.imshow(weight_matrix,
                   labels=dict(x="Tahmin Edilen Sınıf", y="Gerçek Sınıf", color="Ceza Ağırlığı"),
                   title="Quadratic Weight Matrix (5 Sınıf İçin)",
                   color_continuous_scale='Reds')
    
    for i in range(n_classes):
        for j in range(n_classes):
            fig.add_annotation(x=j, y=i, text=f"{weight_matrix[i, j]:.2f}",
                             showarrow=False, 
                             font_color="white" if weight_matrix[i, j] > 0.5 else "black")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.latex(r'''
    \text{Büyük hata (0→2): } 1.0 \text{ vs Küçük hata (0→1): } 0.25 \text{ → 4 kat fark!}
    ''')
    
    # Interactive example based on DATAtab
    st.markdown("### 🎮 Hasta Memnuniyeti Örneği (DATAtab)")
    st.markdown("<strong>İki doktor 75 hastanın tedavi memnuniyetini değerlendiriyor:</strong>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<strong>Confusion Matrix (3x3):</strong>", unsafe_allow_html=True)
        
        # Create 3x3 confusion matrix based on DATAtab example
        confusion_input = np.zeros((3, 3))
        
        categories = ["Memnun Değil", "Nötr", "Memnun"]
        
        for i in range(3):
            st.markdown(f"<strong>Doktor 1: {categories[i]}</strong>", unsafe_allow_html=True)
            cols = st.columns(3)
            for j in range(3):
                with cols[j]:
                    # Default values from DATAtab example
                    default_values = [
                        [15, 3, 0],   # Memnun Değil
                        [5, 20, 2],   # Nötr  
                        [2, 8, 20]    # Memnun
                    ]
                    confusion_input[i, j] = st.number_input(
                        f"→ {categories[j][:4]}", 
                        min_value=0, 
                        value=default_values[i][j],
                        key=f"conf_3x3_{i}_{j}"
                    )
    
    with col2:
        # Calculate both linear and quadratic weighted kappa for 3x3
        def calculate_weighted_kappa(confusion_matrix, weight_type="quadratic"):
            n_classes = confusion_matrix.shape[0]
            
            # Weight matrix
            weight_matrix = np.zeros((n_classes, n_classes))
            for i in range(n_classes):
                for j in range(n_classes):
                    if weight_type == "linear":
                        weight_matrix[i, j] = abs(i - j) / (n_classes - 1)
                    else:  # quadratic
                        weight_matrix[i, j] = (i - j) ** 2 / (n_classes - 1) ** 2
            
            # Observed weighted agreement
            total = np.sum(confusion_matrix)
            observed = 1 - np.sum(weight_matrix * confusion_matrix) / total
            
            # Expected weighted agreement
            row_totals = np.sum(confusion_matrix, axis=1)
            col_totals = np.sum(confusion_matrix, axis=0)
            expected_matrix = np.outer(row_totals, col_totals) / total
            expected = 1 - np.sum(weight_matrix * expected_matrix) / total
            
            if expected == 1.0:
                return 1.0
            else:
                return (observed - expected) / (1 - expected)
        
        total_patients = np.sum(confusion_input)
        
        if total_patients > 0:
            linear_kappa = calculate_weighted_kappa(confusion_input, "linear")
            quadratic_kappa = calculate_weighted_kappa(confusion_input, "quadratic")
            
            st.markdown("<strong>📊 Sonuçlar:</strong>", unsafe_allow_html=True)
            st.metric("Toplam Hasta", f"{int(total_patients)}")
            st.metric("Linear Weighted Kappa", f"{linear_kappa:.3f}")
            st.metric("Quadratic Weighted Kappa", f"{quadratic_kappa:.3f}")
            
            # Show confusion matrix
            fig = px.imshow(confusion_input,
                           labels=dict(x="Doktor 2", y="Doktor 1"),
                           x=categories, y=categories,
                           title="Hasta Memnuniyeti Confusion Matrix")
            
            for i in range(3):
                for j in range(3):
                    fig.add_annotation(x=j, y=i, text=str(int(confusion_input[i, j])),
                                     showarrow=False, font_color="white")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Weight matrix visualization
            st.markdown("<strong>⚖️ Quadratic Weight Matrix:</strong>", unsafe_allow_html=True)
            weight_matrix = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    weight_matrix[i, j] = (i - j) ** 2 / (3 - 1) ** 2
            
            weight_df = pd.DataFrame(weight_matrix)
            weight_df.index = categories
            weight_df.columns = categories
            st.dataframe(weight_df.round(3), use_container_width=True)
            
            # Detailed agreement analysis
            perfect_agreement = confusion_input[0,0] + confusion_input[1,1] + confusion_input[2,2]
            one_step_errors = confusion_input[0,1] + confusion_input[1,0] + confusion_input[1,2] + confusion_input[2,1]
            two_step_errors = confusion_input[0,2] + confusion_input[2,0]
            
            st.markdown("<strong>🔍 Uyuşma Analizi:</strong>", unsafe_allow_html=True)
            st.metric("Tam Uyuşma", f"{int(perfect_agreement)} hasta")
            st.metric("1 Adım Hata", f"{int(one_step_errors)} hasta")
            st.metric("2 Adım Hata", f"{int(two_step_errors)} hasta")
    
    st.markdown("""
    ### 📋 Adım Adım Hesaplama Örneği
    
    DATAtab örneğinden (75 hasta, 3 kategori):
    """)
    
    st.latex(r'''
    \text{1. Weight Matrix oluştur: } w_{ij} = \frac{(i-j)^2}{(N-1)^2}
    ''')
    
    st.latex(r'''
    \text{2. Observed Agreement: } P_o^w = 1 - \frac{\sum w_{ij} \cdot O_{ij}}{n}
    ''')
    
    st.latex(r'''
    \text{3. Expected Agreement: } P_e^w = 1 - \frac{\sum w_{ij} \cdot E_{ij}}{n}
    ''')
    
    st.latex(r'''
    \text{4. Weighted Kappa: } \kappa_w = \frac{P_o^w - P_e^w}{1 - P_e^w}
    ''')
    
    st.markdown("""
    <div class="fun-fact">
    <h3>🌟 Weighted Kappa'nın Kullanım Alanları</h3>
    <p>• <strong>Medical Assessment</strong>: Hastalık severity, tedavi response<br>
    • <strong>Education</strong>: Öğrenci performans değerlendirmesi<br>
    • <strong>Psychology</strong>: Likert scale anketler<br>
    • <strong>Quality Control</strong>: Ürün kalite sınıflandırması<br>
    • <strong>Kaggle</strong>: Ordinal prediction competitions</p>
    </div>
    """, unsafe_allow_html=True)

def log_loss_page():
    """Log Loss explanation page"""
    st.markdown("# 📉 Log Loss")
    
    st.markdown("""
    <div class="explanation-box">
    <h2>🤔 Log Loss Nedir?</h2>
    <p><strong>Basit açıklama:</strong> Modelin ne kadar "emin" olduğunu ölçer! 
    Yanlış tahminlerde çok emin olmayı ağır cezalandırır.</p>
    
    <p><strong>Gerçek hayat analojisi:</strong> Hava durumu tahmini:<br>
    • "Yağmur %90 ihtimal" deyip güneşli çıkarsa → Büyük ceza<br>
    • "Yağmur %55 ihtimal" deyip güneşli çıkarsa → Küçük ceza<br>
    Aşırı güven kötü, makul şüphe iyi!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Formula explanation
    st.markdown("### 📐 Formül")
    
    st.markdown("**Binary Classification:**")
    st.latex(r'''
    \text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]
    ''')
    
    st.markdown("**Multi-class Classification:**")
    st.latex(r'''
    \text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} y_{ij} \log(p_{ij})
    ''')
    
    st.markdown("""
    **🧠 Formülün Mantığı:**
    
    Bu formül **modelin güven seviyesini** cezalandırmak için tasarlandı:
    
    - **yi**: Gerçek label (0 veya 1 binary'de)
    - **pi**: Model'in tahmin ettiği probability
    - **log(pi)**: Logaritmik ceza → Düşük probability'lerde çok büyür!
    - **Negatif işaret**: Loss'u pozitif yapmak için
    
    **💡 Neden Logaritma?**
    """)
    
    st.latex(r'''
    p = 0.9 \rightarrow \log(0.9) = -0.046 \rightarrow \text{Küçük ceza}
    ''')
    
    st.latex(r'''
    p = 0.1 \rightarrow \log(0.1) = -1.0 \rightarrow \text{Büyük ceza}
    ''')
    
    st.latex(r'''
    p = 0.01 \rightarrow \log(0.01) = -2.0 \rightarrow \text{Çok büyük ceza!}
    ''')
    
    st.markdown("""
    
    **Aşırı güvenli yanlış tahminler çok ağır cezalandırılır!**
    """)
    
    st.markdown("""
    <div style="background: #f3e5f5; padding: 1rem; border-radius: 8px; border-left: 4px solid #E91E63;">
    <strong>📚 Tarihçe:</strong> Log Loss aslında **information theory**'den gelir. Claude Shannon'ın 1948'deki 
    "bilgi entropisi" kavramına dayanır. Bir olayın gerçekleşme probability'si düştükçe, 
    o olayın "information content" artar. Yani nadir olayları yanlış tahmin etmek daha fazla "bilgi kaybı" demektir!
    Bu yüzden Neural Network'ler de Cross-Entropy Loss (aynı şey) kullanır.
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive probability demo
    st.markdown("### 🎯 Probability Confidence Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Senaryo Kurgusu:**")
        true_class = st.selectbox("Gerçek sınıf", [0, 1], index=1)
        predicted_prob = st.slider("Model'in sınıf 1 için tahmini (%)", 0.0, 100.0, 80.0) / 100
        
        # Calculate log loss for this single prediction
        epsilon = 1e-15  # To avoid log(0)
        if true_class == 1:
            single_log_loss = -np.log(max(predicted_prob, epsilon))
        else:
            single_log_loss = -np.log(max(1 - predicted_prob, epsilon))
        
        st.metric("Bu Tahmin İçin Log Loss", f"{single_log_loss:.3f}")
    
    with col2:
        # Visualization of how confidence affects loss
        prob_range = np.linspace(0.01, 0.99, 100)
        
        # Log loss for correct prediction (true class = 1)
        correct_loss = -np.log(prob_range)
        # Log loss for incorrect prediction (true class = 0)
        incorrect_loss = -np.log(1 - prob_range)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prob_range, y=correct_loss, 
                               name='Doğru Tahmin (True=1)', 
                               line=dict(color='green')))
        fig.add_trace(go.Scatter(x=prob_range, y=incorrect_loss, 
                               name='Yanlış Tahmin (True=0)', 
                               line=dict(color='red')))
        
        # Add current prediction point
        current_loss = single_log_loss
        fig.add_trace(go.Scatter(x=[predicted_prob], y=[current_loss],
                               mode='markers', marker=dict(size=15, color='blue'),
                               name='Sizin Tahmininiz'))
        
        fig.update_layout(
            title="Confidence vs Log Loss",
            xaxis_title="Predicted Probability",
            yaxis_title="Log Loss",
            yaxis=dict(range=[0, 5])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Multi-class demo
    st.markdown("### 🎮 Multi-Class Log Loss Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**3 Sınıflı Bir Problem:**")
        st.markdown("Animal Classification: Kedi, Köpek, Kuş")
        
        true_animal = st.selectbox("Gerçek hayvan", ["Kedi (0)", "Köpek (1)", "Kuş (2)"])
        true_class_idx = int(true_animal.split("(")[1].split(")")[0])
        
        st.markdown("**Model'in tahmin olasılıkları:**")
        prob_cat = st.slider("Kedi olasılığı (%)", 0.0, 100.0, 33.3) / 100
        prob_dog = st.slider("Köpek olasılığı (%)", 0.0, 100.0, 33.3) / 100
        prob_bird = st.slider("Kuş olasılığı (%)", 0.0, 100.0, 33.4) / 100
        
        # Normalize probabilities
        total_prob = prob_cat + prob_dog + prob_bird
        if total_prob > 0:
            prob_cat /= total_prob
            prob_dog /= total_prob
            prob_bird /= total_prob
        
        probs = [prob_cat, prob_dog, prob_bird]
        
    with col2:
        # Calculate log loss
        epsilon = 1e-15
        true_prob = max(probs[true_class_idx], epsilon)
        log_loss_value = -np.log(true_prob)
        
        st.markdown("**📊 Sonuçlar:**")
        st.metric("Normalize Edilmiş Olasılıklar", 
                 f"Kedi: {prob_cat:.2f}, Köpek: {prob_dog:.2f}, Kuş: {prob_bird:.2f}")
        st.metric("Log Loss", f"{log_loss_value:.3f}")
        
        # Interpretation
        if log_loss_value < 0.5:
            interpretation = "🎯 Çok Güvenli Tahmin!"
        elif log_loss_value < 1.0:
            interpretation = "😊 İyi Tahmin"
        elif log_loss_value < 2.0:
            interpretation = "😐 Orta Tahmin"
        else:
            interpretation = "😱 Kötü Tahmin"
        
        st.markdown(f"**Değerlendirme:** {interpretation}")
        
        # Probability bar chart
        fig = go.Figure(data=[
            go.Bar(x=['Kedi', 'Köpek', 'Kuş'], 
                  y=[prob_cat, prob_dog, prob_bird],
                  marker_color=['lightcoral' if i == true_class_idx else 'lightblue' 
                               for i in range(3)])
        ])
        fig.update_layout(title="Model Tahmin Olasılıkları", 
                         yaxis_title="Olasılık")
        st.plotly_chart(fig, use_container_width=True)
    
    # Real data example
    st.markdown("### 🔬 Gerçek Veri Örneği")
    y_true, y_pred, y_pred_proba = create_sample_data()
    
    # Calculate log loss
    real_log_loss = log_loss(y_true, y_pred_proba)
    st.metric("Örnek Veri Log Loss", f"{real_log_loss:.3f}")
    
    # Show some predictions
    sample_indices = np.random.choice(len(y_true), 5, replace=False)
    sample_df = pd.DataFrame({
        'Gerçek Sınıf': y_true[sample_indices],
        'Tahmin Sınıf': y_pred[sample_indices],
        'Sınıf 0 Prob': y_pred_proba[sample_indices, 0],
        'Sınıf 1 Prob': y_pred_proba[sample_indices, 1],
        'Sınıf 2 Prob': y_pred_proba[sample_indices, 2]
    })
    
    st.markdown("<strong>Örnek Tahminler:</strong>", unsafe_allow_html=True)
    st.dataframe(sample_df, use_container_width=True)
    
    st.markdown("""
    <div class="fun-fact">
    <h3>🎯 Log Loss'un Özellikleri</h3>
    <p>• Sadece doğru/yanlış değil, güven seviyesini de ölçer<br>
    • Aşırı güvenli yanlış tahminleri ağır cezalandırır<br>
    • Probability calibration için mükemmel<br>
    • Neural network training'de cross-entropy olarak kullanılır</p>
    </div>
    """, unsafe_allow_html=True)

def focal_loss_page():
    """Focal Loss explanation page"""
    st.markdown("# 🔍 Focal Loss")
    
    st.markdown("""
    <div class="explanation-box">
    <h2>🤔 Focal Loss Nedir?</h2>
    <p><strong>Basit açıklama:</strong> Log Loss'un akıllı versiyonu! 
    Kolay örnekleri ihmal edip zor örneklere odaklanır. Özellikle imbalanced data'da çok işe yarar.</p>
    
    <p><strong>Gerçek hayat analojisi:</strong> Sınıfta ders anlatan öğretmen:<br>
    • Anlayan öğrencilerle az zaman geçirir<br>
    • Anlamayan öğrencilere daha çok zaman ayırır<br>
    Focal Loss da böyle çalışır - zor örneklere daha çok odaklanır!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Formula explanation
    st.markdown("### 📐 Formül")
    
    st.latex(r'''
    \text{Focal Loss} = -\alpha (1-p_t)^\gamma \log(p_t)
    ''')
    
    st.markdown("**Binary classification için açılım:**")
    st.latex(r'''
    FL(p_t) = \begin{cases}
    -\alpha (1-p)^\gamma \log(p) & \text{if } y = 1 \\
    -(1-\alpha) p^\gamma \log(1-p) & \text{if } y = 0
    \end{cases}
    ''')
    
    st.markdown("""
    **🧠 Formülün Mantığı:**
    
    Bu formül **Log Loss'u akıllıca modifiye eder** zor örneklere odaklanmak için:
    
    - **pt**: Doğru sınıf için tahmin edilen probability
    - **α (alpha)**: Sınıf dengeleme faktörü (0.25 tipik değer)
    - **γ (gamma)**: Focusing parameter (2.0 tipik değer)
    - **(1-pt)^γ**: Modulating factor → **Bu büyü burada!**
    
    **💡 (1-pt)^γ Büyüsü:**
    """)
    
    st.latex(r'''
    p_t = 0.9 \text{ (kolay örnek)} \rightarrow (1-0.9)^2 = 0.01 \rightarrow \text{Loss 100'de 1'ine iner!}
    ''')
    
    st.latex(r'''
    p_t = 0.6 \text{ (orta örnek)} \rightarrow (1-0.6)^2 = 0.16 \rightarrow \text{Loss 6'da 1'ine iner}
    ''')
    
    st.latex(r'''
    p_t = 0.2 \text{ (zor örnek)} \rightarrow (1-0.2)^2 = 0.64 \rightarrow \text{Loss neredeyse aynı kalır}
    ''')
    
    st.markdown("""
    
    **Kolay örnekler ihmal edilir, zor örneklere odaklanılır!**
    """)
    
    st.markdown("""
    <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; border-left: 4px solid #4CAF50;">
    <strong>📚 Tarihçe:</strong> Tsung-Yi Lin ve ekibi tarafından 2017'de **RetinaNet** paper'ında tanıtıldı. 
    Object detection'da **class imbalance** problemi vardı: background pixels çok, object pixels az. 
    Model kolay background'ları öğrenip object'leri ihmal ediyordu. Focal Loss bunu çözdü ve 
    one-stage detector'ları two-stage kadar başarılı hale getirdi!
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive parameter exploration
    st.markdown("### 🎮 Parameter Etkisini Keşfet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Parametreleri Ayarla:**")
        alpha = st.slider("Alpha (α) - Sınıf Ağırlığı", 0.1, 2.0, 1.0, 0.1)
        gamma = st.slider("Gamma (γ) - Odaklanma Gücü", 0.0, 5.0, 2.0, 0.5)
        
        st.markdown("**Test Senaryosu:**")
        true_class = st.selectbox("Gerçek sınıf", [0, 1], index=1, key="focal_true")
        predicted_prob = st.slider("Tahmin olasılığı", 0.01, 0.99, 0.7, key="focal_prob")
    
    with col2:
        # Calculate focal loss vs standard log loss
        epsilon = 1e-15
        
        if true_class == 1:
            p = max(predicted_prob, epsilon)
        else:
            p = max(1 - predicted_prob, epsilon)
        
        # Standard log loss
        log_loss_val = -np.log(p)
        
        # Focal loss
        focal_loss_val = -alpha * ((1 - p) ** gamma) * np.log(p)
        
        st.markdown("**📊 Karşılaştırma:**")
        st.metric("Standard Log Loss", f"{log_loss_val:.3f}")
        st.metric("Focal Loss", f"{focal_loss_val:.3f}")
        st.metric("Focal/Log Ratio", f"{focal_loss_val/log_loss_val:.2f}")
        
        # Difficulty assessment
        if p > 0.8:
            difficulty = "🟢 Kolay Örnek"
        elif p > 0.6:
            difficulty = "🟡 Orta Örnek"
        else:
            difficulty = "🔴 Zor Örnek"
        
        st.markdown(f"**Örnek Zorluğu:** {difficulty}")
    
    # Comparative visualization
    st.markdown("### 📊 Focal Loss vs Log Loss Karşılaştırması")
    
    prob_range = np.linspace(0.01, 0.99, 100)
    log_losses = -np.log(prob_range)
    focal_losses = -alpha * ((1 - prob_range) ** gamma) * np.log(prob_range)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_range, y=log_losses, 
                           name='Standard Log Loss', 
                           line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=prob_range, y=focal_losses, 
                           name=f'Focal Loss (α={alpha}, γ={gamma})', 
                           line=dict(color='red')))
    
    # Add current prediction point
    current_p = predicted_prob if true_class == 1 else (1 - predicted_prob)
    current_log = -np.log(max(current_p, epsilon))
    current_focal = -alpha * ((1 - current_p) ** gamma) * np.log(max(current_p, epsilon))
    
    fig.add_trace(go.Scatter(x=[current_p], y=[current_log],
                           mode='markers', marker=dict(size=12, color='blue'),
                           name='Sizin Log Loss'))
    fig.add_trace(go.Scatter(x=[current_p], y=[current_focal],
                           mode='markers', marker=dict(size=12, color='red'),
                           name='Sizin Focal Loss'))
    
    fig.update_layout(
        title="Loss Comparison: Easy vs Hard Examples",
        xaxis_title="Predicted Probability (for True Class)",
        yaxis_title="Loss Value",
        annotations=[
            dict(x=0.9, y=0.5, text="Kolay Örnekler<br>(Yüksek p)", 
                 showarrow=False, bgcolor="lightgreen", opacity=0.7),
            dict(x=0.1, y=3, text="Zor Örnekler<br>(Düşük p)", 
                 showarrow=False, bgcolor="lightcoral", opacity=0.7)
        ]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Imbalanced dataset demo
    st.markdown("### ⚖️ Imbalanced Dataset Senaryosu")
    
    st.markdown("""
    <div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #4169E1;">
    <strong>🎯 Senaryo:</strong> Bir hastanede kanser tespiti yapan AI modeli var.<br>
    • <strong>Çoğunluk sınıfı</strong>: Sağlıklı hastalar (tahmin etmesi kolay)<br>
    • <strong>Azınlık sınıfı</strong>: Kanserli hastalar (tahmin etmesi zor ve kritik!)
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<strong>🏥 Dataset Kompozisyonu:</strong>", unsafe_allow_html=True)
        majority_class_size = st.slider("Sağlıklı hastalar (kolay örnekler)", 500, 2000, 1000)
        minority_class_size = st.slider("Kanserli hastalar (zor örnekler)", 10, 200, 50)
        
        # Simulate some predictions
        np.random.seed(42)
        # Majority class predictions (easier to predict)
        maj_probs = np.random.beta(8, 2, majority_class_size)  # High confidence
        # Minority class predictions (harder to predict)  
        min_probs = np.random.beta(3, 3, minority_class_size)  # Lower confidence
        
        # Calculate losses
        maj_log_losses = -np.log(np.clip(maj_probs, epsilon, 1-epsilon))
        min_log_losses = -np.log(np.clip(min_probs, epsilon, 1-epsilon))
        
        maj_focal_losses = -alpha * ((1 - maj_probs) ** gamma) * np.log(np.clip(maj_probs, epsilon, 1-epsilon))
        min_focal_losses = -alpha * ((1 - min_probs) ** gamma) * np.log(np.clip(min_probs, epsilon, 1-epsilon))
        
    with col2:
        # Show average losses
        avg_maj_log = np.mean(maj_log_losses)
        avg_min_log = np.mean(min_log_losses)
        avg_maj_focal = np.mean(maj_focal_losses)
        avg_min_focal = np.mean(min_focal_losses)
        
        st.markdown("<strong>📊 Ortalama Loss Değerleri (Model Ne Kadar Zorlanıyor?):</strong>", unsafe_allow_html=True)
        
        loss_comparison_df = pd.DataFrame({
            'Loss Türü': ['Log Loss', 'Focal Loss'],
            'Sağlıklı (Kolay)': [f"{avg_maj_log:.3f}", f"{avg_maj_focal:.3f}"],
            'Kanserli (Zor)': [f"{avg_min_log:.3f}", f"{avg_min_focal:.3f}"]
        })
        
        st.dataframe(loss_comparison_df, use_container_width=True)
        
        st.markdown("""
        <div style="background: #e8f5e8; padding: 0.8rem; border-radius: 6px;">
        <strong>📖 Tablo Nasıl Okunur:</strong><br>
        • <strong>Düşük sayı</strong>: Model bu grubu kolay tahmin ediyor<br>
        • <strong>Yüksek sayı</strong>: Model bu grubu zor tahmin ediyor<br>
        • Focal Loss kolay örnekleri "ihmal ediyor" (sayı düşüyor)
        </div>
        """, unsafe_allow_html=True)
        
        # Show the focus shift with better explanation
        total_log_loss = (majority_class_size * avg_maj_log + minority_class_size * avg_min_log) / (majority_class_size + minority_class_size)
        total_focal_loss = (majority_class_size * avg_maj_focal + minority_class_size * avg_min_focal) / (majority_class_size + minority_class_size)
        
        st.markdown("<strong>🎯 Genel Performans Analizi:</strong>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Toplam Log Loss", f"{total_log_loss:.3f}", help="Tüm örneklerin ortalama loss'u")
        with col_b:
            st.metric("Toplam Focal Loss", f"{total_focal_loss:.3f}", help="Focal loss ile ortalama loss")
        
        # Ratio showing focus shift
        focus_ratio = (avg_min_focal / avg_maj_focal) / (avg_min_log / avg_maj_log)
        st.metric("Odaklanma Gücü", f"{focus_ratio:.2f}x", 
                 help="Focal Loss'un zor örneklere ne kadar daha fazla odaklandığı")
        
        if focus_ratio > 1.5:
            st.success("✅ Focal Loss zor örneklere güçlü şekilde odaklanıyor!")
        elif focus_ratio > 1.2:
            st.info("ℹ️ Focal Loss orta seviyede odaklanma sağlıyor")
        else:
            st.warning("⚠️ Focal Loss çok az odaklanma sağlıyor")
    
    # Distribution visualization with clear explanation
    st.markdown("### 📈 Loss Dağılımları Analizi")
    
    st.markdown("""
    <div style="background: #fff3e0; padding: 1rem; border-radius: 8px; border-left: 4px solid #FF9800;">
    <strong>📊 Bu Grafik Neyi Gösteriyor?</strong><br>
    Her çubuk, o loss değerinde kaç tane hasta olduğunu gösteriyor.<br>
    • <strong>Sola yakın</strong> (düşük loss): Model emin, doğru tahmin<br>
    • <strong>Sağa yakın</strong> (yüksek loss): Model kararsız, yanlış tahmin<br>
    • <strong>Focal Loss'un gücü</strong>: Kolay örnekleri sola itiyor (ihmal ediyor)
    </div>
    """, unsafe_allow_html=True)
    
    # Create side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=maj_log_losses, name='Sağlıklı (Kolay)', 
                                   opacity=0.7, nbinsx=20, marker_color='lightblue'))
        fig1.add_trace(go.Histogram(x=min_log_losses, name='Kanserli (Zor)', 
                                   opacity=0.7, nbinsx=20, marker_color='lightcoral'))
        
        fig1.update_layout(title="📊 Log Loss Dağılımı", 
                          xaxis_title="Loss Değeri", 
                          yaxis_title="Hasta Sayısı",
                          barmode='overlay',
                          height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("""
        <strong>🔍 Log Loss'ta Ne Görüyoruz?</strong><br>
        • Mavi (sağlıklı): Düşük loss, kolay tahmin<br>
        • Kırmızı (kanserli): Yüksek loss, zor tahmin<br>
        • Model her iki gruba da eşit önem veriyor
        """, unsafe_allow_html=True)
    
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=maj_focal_losses, name='Sağlıklı (Kolay)', 
                                   opacity=0.7, nbinsx=20, marker_color='lightgreen'))
        fig2.add_trace(go.Histogram(x=min_focal_losses, name='Kanserli (Zor)', 
                                   opacity=0.7, nbinsx=20, marker_color='orange'))
        
        fig2.update_layout(title="🎯 Focal Loss Dağılımı", 
                          xaxis_title="Loss Değeri", 
                          yaxis_title="Hasta Sayısı",
                          barmode='overlay',
                          height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        <strong>🎯 Focal Loss'ta Ne Görüyoruz?</strong><br>
        • Yeşil (sağlıklı): Çok düşük loss, ihmal ediliyor<br>
        • Turuncu (kanserli): Loss değişmedi, odaklanılıyor<br>
        • Model zor örneklere odaklanmaya başladı!
        """, unsafe_allow_html=True)
    
    # Summary comparison
    st.markdown("### 🔬 Karşılaştırma Özeti")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        kolay_azalma = ((avg_maj_log - avg_maj_focal) / avg_maj_log) * 100
        st.metric("Kolay Örnekler", f"{kolay_azalma:.1f}% azaldı", 
                 delta=f"Loss {avg_maj_log:.3f}→{avg_maj_focal:.3f}")
    
    with insight_col2:
        zor_azalma = ((avg_min_log - avg_min_focal) / avg_min_log) * 100
        st.metric("Zor Örnekler", f"{zor_azalma:.1f}% azaldı", 
                 delta=f"Loss {avg_min_log:.3f}→{avg_min_focal:.3f}")
    
    with insight_col3:
        if kolay_azalma > zor_azalma * 2:
            st.success("✅ Mükemmel! Focal Loss kolay örnekleri daha çok azalttı")
        elif kolay_azalma > zor_azalma:
            st.info("ℹ️ İyi! Kolay örnekler daha fazla azaldı")
        else:
            st.warning("⚠️ Focal Loss beklenen etkiyi göstermiyor")
    
    st.markdown("""
    <div class="fun-fact">
    <h3>🌟 Focal Loss'un Süper Güçleri</h3>
    <p>• Imbalanced dataset'lerde mükemmel<br>
    • Zor örneklere otomatik odaklanma<br>
    • Object detection'da devrim yarattı<br>
    • RetinaNet gibi modern modellerde kullanılır<br>
    • Hyperparameter tuning ile çok esnek</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Bir metrik seçin:",
        [
            "🏠 Ana Sayfa",
            "⚖️ Balanced Accuracy", 
            "🔗 Matthews Correlation Coefficient",
            "🤝 Cohen's Kappa",
            "⚖️ Quadratic Weighted Kappa",
            "📉 Log Loss",
            "🔍 Focal Loss"
        ]
    )
    
    # Page routing
    if page == "🏠 Ana Sayfa":
        homepage()
    elif page == "⚖️ Balanced Accuracy":
        balanced_accuracy_page()
    elif page == "🔗 Matthews Correlation Coefficient":
        matthews_correlation_page()
    elif page == "🤝 Cohen's Kappa":
        cohens_kappa_page()
    elif page == "⚖️ Quadratic Weighted Kappa":
        quadratic_weighted_kappa_page()
    elif page == "📉 Log Loss":
        log_loss_page()
    elif page == "🔍 Focal Loss":
        focal_loss_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>🎓 ML Metrics Akademisi</p>
    <p>Eğlenceli öğrenme deneyimi</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 