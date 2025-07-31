# Water Potability Classification

Bu proje, içme suyu örneklerinin içilebilirlik durumunu tahmin etmek için makine öğrenimi yöntemleri kullanmaktadır. Veri setindeki eksik veriler temizlenmiş, çeşitli görselleştirmeler yapılmış ve iki temel model (Decision Tree ve Random Forest) eğitilmiştir. Son olarak modelin performansını artırmak için hiperparametre optimizasyonu uygulanmıştır.

Veri seti: https://www.kaggle.com/datasets/adityakadiwal/water-potability

## Kullanılan Teknolojiler
- Python 3
- Pandas, NumPy, Seaborn, Matplotlib, Plotly
- Scikit-learn (Decision Tree, Random Forest, RandomizedSearchCV)

## Proje Adımları
1. Veri keşfi ve görselleştirme
2. Eksik verilerin sınıfa özel ortalamalarla doldurulması
3. Min-Max normalizasyonu ile veri ölçeklendirme
4. Modellerin eğitimi ve değerlendirilmesi
5. Hiperparametre ayarlaması ile model performansının artırılması

## Kurulum ve Çalıştırma

```bash
# Gerekli kütüphaneleri yükleyin
pip install pandas numpy scikit-learn matplotlib seaborn plotly missingno

