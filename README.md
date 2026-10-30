# 🏡 California House Price Prediction App

An end-to-end Machine Learning web application that predicts housing prices in California using Linear, Ridge, and Lasso Regression. Built with Python, Scikit-Learn, and Streamlit.

---

## 🚀 Live Demo
*(If you deploy it on Streamlit Community Cloud, add your link here!)*  
https://house-price-prediction-q8jgiappt4lywvqycgjfyzq.streamlit.app/

---

## 📌 Project Overview
This project explores the famous California Housing dataset to predict median house values based on features like median income, house age, average rooms, population, and geographical location. It compares multiple linear machine learning models and wraps them into an interactive web dashboard for real-time predictions.

---

## 🛠️ Tech Stack & Libraries
* **Python** (Core programming language)
* **Pandas & NumPy** (Data manipulation and analysis)
* **Scikit-Learn** (Model training, scaling, and evaluation)
* **Matplotlib & Seaborn** (Exploratory Data Analysis and visualization)
* **Streamlit** (Interactive web app interface)

---

## 📊 Machine Learning Pipeline
1. **Data Collection & EDA:** Loaded the California Housing dataset, analyzed feature distributions, and checked correlations using heatmaps.
2. **Preprocessing & Scaling:** Handled feature scaling using `StandardScaler` to ensure features with large ranges (like population) don't overpower smaller ones (like median income).
3. **Model Training:** Trained and evaluated three models:
   * **Linear Regression** (Baseline model)
   * **Ridge Regression** (L2 Regularization to prevent overfitting)
   * **Lasso Regression** (L1 Regularization for feature selection)
4. **Evaluation Metrics:** Evaluated models using **MAE** (Mean Absolute Error), **RMSE** (Root Mean Squared Error), and **R² Score**.

---

## 📂 Project Structure
```text
House-Price-Prediction/
│
├── models/               # Saved model files (.pkl)
│   ├── linear_model.pkl
│   ├── ridge_model.pkl
│   └── scaler.pkl
│
├── eda.py                # Exploratory Data Analysis script
├── train_models.py       # Script to train and save models
├── app.py                # Streamlit web application
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
