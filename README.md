# 🏠 Boston House Price Prediction

## 📌 Project Overview

This project focuses on predicting **house prices in Boston** using classical **Machine Learning regression techniques**. The goal is to understand how different socio-economic and geographic factors influence housing prices and to build a robust predictive model.

The dataset is a well-known benchmark in machine learning, often used to demonstrate **end-to-end regression pipelines**, including data preprocessing, feature scaling, model training, evaluation, and deployment readiness.

---

## 🎯 Problem Statement

Given multiple features related to housing conditions (crime rate, number of rooms, accessibility, tax rate, etc.), predict the **median house price**.

* **Problem Type**: Supervised Learning
* **Task**: Regression
* **Target Variable**: `price` (formerly `MEDV`)

---

## 📂 Dataset Description

The dataset contains **506 rows and 14 columns**.

### 🔑 Key Features

| Feature | Description                                          |
| ------- | ---------------------------------------------------- |
| CRIM    | Crime rate per capita                                |
| ZN      | Residential land zoned                               |
| INDUS   | Proportion of non-retail business acres              |
| NOX     | Nitric oxide concentration                           |
| RM      | Average number of rooms                              |
| AGE     | Proportion of owner-occupied units built before 1940 |
| DIS     | Distance to employment centers                       |
| RAD     | Accessibility to highways                            |
| TAX     | Property tax rate                                    |
| PTRATIO | Pupil-teacher ratio                                  |
| B       | Proportion of Black population                       |
| LSTAT   | % of lower status population                         |
| CHAS    | Charles River dummy variable                         |

**Target**:

* `price` – Median value of owner-occupied homes

---

## ⚙️ Tech Stack

* **Programming Language**: Python
* **Libraries**:

  * NumPy
  * Pandas
  * Matplotlib / Seaborn
  * Scikit-learn
* **Environment**: Jupyter Notebook

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Loading & Exploration

* Checked shape, null values, and data types
* Renamed target column `MEDV → price`
* Performed correlation analysis

### 2️⃣ Feature Engineering & Preprocessing

* Split data into **features (X)** and **target (y)**
* Train-test split (70%-30%)
* Applied **StandardScaler** on input features only

> ⚠️ *Target variable was intentionally NOT scaled to avoid distorted evaluation metrics.*

### 3️⃣ Model Training

* Model Used: **Linear Regression**
* Trained using scaled training features

### 4️⃣ Model Evaluation

Used multiple regression metrics:

* **R² Score**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Error (MAE)**

### 5️⃣ Model Serialization

* Model saved using **pickle** for reuse and deployment

---

## 📊 Results

| Metric   | Value                               |
| -------- | ----------------------------------- |
| R² Score | Indicates model explanatory power   |
| RMSE     | Measures prediction error magnitude |
| MAE      | Average absolute prediction error   |

The model performs well for a linear baseline and serves as a strong foundation for advanced models.

---

## 🚨 Common Pitfalls Addressed

* ❌ **Avoided scaling the target variable**
* ❌ **Prevented data leakage during scaling**
* ❌ **Ensured predictions were not compared against the same data**
* ❌ **Corrected pickling of trained model instead of untrained instance**

---

## 📈 Visualization

* Scatter plot of **Actual vs Predicted Prices**
* Residual distribution analysis
* Feature correlation heatmap

---

## 🚀 Future Improvements

* Implement **Polynomial Regression**
* Apply **Ridge, Lasso, ElasticNet**
* Try **Tree-based models (RandomForest, XGBoost)**
* Deploy model using **Flask / FastAPI**
* Add **cross-validation** and **hyperparameter tuning**

---

## 🧪 How to Run the Project

```bash
# Clone the repository
git clone <repo-url>

# Navigate to folder
cd boston-house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook
```

---

## 👤 Author

**Saurav Pawar**
Data Science & Machine Learning Enthusiast

---

## 📜 License

This project is for educational and learning purposes.
