# 📰 Fake News Detection App

This project is a **Streamlit-based web application** that detects whether a news headline or content is real or fake using **Natural Language Processing (NLP)** and **Machine Learning**. It uses TF-IDF vectorization and is trained on a labeled fake news dataset.

---

## 🚀 Features

- Streamlit web interface for easy usage
- Preprocessing with stemming and stopword removal
- TF-IDF vectorization of text data
- Multiple ML models trained (Logistic Regression, Random Forest, Linear SVM)
- Automatically selects and saves the best model
- Persistent model saved using `joblib`
- Real-time prediction of user input news text

---

## 🧠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas / NumPy
- NLTK (stopwords, stemming)
- Joblib (model saving/loading)

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
