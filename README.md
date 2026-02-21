# 🌾 Agro Guide AI: Precision Crop Recommendation System

**Agro Guide AI** is a full-stack machine learning web application designed to help farmers optimize their yield. By analyzing soil composition and environmental factors, the system provides data-driven crop recommendations using a trained Random Forest model.

---

## 🚀 Key Features
* **Smart Crop Prediction:** Recommends the best crop to grow based on Nitrogen (N), Phosphorous (P), Potassium (K), and pH levels.
* **Climate-Aware:** Factors in real-time environmental data like Temperature, Humidity, and Rainfall.
* **FastAPI Backend:** High-performance asynchronous API for real-time model inference.
* **Modern UI:** Clean, responsive web interface for easy data input and result visualization.

---

## 🛠️ Tech Stack
* **Frontend:** HTML5, CSS3 (Tailwind CSS), JavaScript
* **Backend:** FastAPI (Python)
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Model:** Random Forest Classifier (Accuracy: ~99%)
* **Deployment:** Uvicorn

---

## 📂 Project Structure
```text
Agro-Guide-AI/
├── main.py              # FastAPI application & API routes
├── model.pkl            # Trained ML model (Pickle file)
├── requirements.txt     # Project dependencies             
└── index.html           # HTML templates (index.html)
