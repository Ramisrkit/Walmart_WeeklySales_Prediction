# Walmart_WeeklySales_Prediction
# 🏬 Walmart Weekly Sales Prediction

## 📌 Project Overview

This project focuses on **predicting weekly sales for Walmart stores** using machine learning techniques. Accurate sales forecasting helps retailers improve **inventory planning, demand forecasting, and promotional strategies**.

The notebook **`Walmart_WeeklySales_Prediction.ipynb`** covers the complete end‑to‑end pipeline: data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter tuning, and evaluation.

---

## 🎯 Objectives

* Predict **Weekly_Sales** accurately
* Understand key factors affecting sales
* Apply ensemble machine learning models
* Reduce overfitting using proper validation
* Visualize actual vs predicted sales

---

## 📂 Dataset

* **Source:** Walmart Sales Dataset (Kaggle)
* **Type:** Time‑based tabular data
* **Target Variable:** `Weekly_Sales`
* **Features include:**

  * Store
  * Dept
  * Date
  * Temperature
  * Fuel_Price
  * CPI
  * Unemployment
  * Holiday_Flag

---

## ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit‑learn
* Matplotlib
* Seaborn

---

## 🧠 Models Implemented

### 🔹 Random Forest Regressor

Random Forest was selected because:

* Handles non‑linear relationships effectively
* Robust to outliers
* Works well with tabular business data
* Reduces variance using bagging

(Optional models such as Linear Regression, Decision Tree, or Gradient Boosting can be added for comparison.)

---

## 🔧 Hyperparameter Tuning

Hyperparameters were optimized using **GridSearchCV / RandomizedSearchCV** with cross‑validation to balance **bias and variance**.

### Example Parameter Grid

```python
parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8, 10],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5],
    'max_features': ['sqrt', 'log2']
}
```

---

## 📊 Model Evaluation

### Metrics Used

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

### Visualizations

* Actual vs Predicted Sales Scatter Plot
* Feature Importance Plot

These plots help verify **model generalization and prediction quality**.

---

## 📈 Results & Observations

* Ensemble models significantly outperform baseline models
* Random Forest shows strong generalization
* Seasonal and economic indicators impact weekly sales
* Predictions closely follow actual sales trends

---

## 🚀 How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/your-username/Walmart-Weekly-Sales-Prediction.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Open the notebook

```bash
jupyter notebook Walmart_WeeklySales_Prediction.ipynb
```

---

## 📌 Future Improvements

* Add time‑series models (ARIMA, Prophet)
* Use XGBoost / LightGBM for better accuracy
* Perform store‑wise forecasting
* Deploy model using Streamlit or Flask
* Include holiday‑specific sales impact analysis

---

## 🏁 Conclusion

This project demonstrates how machine learning can be effectively applied to **retail sales forecasting**. The Random Forest model provides reliable predictions and valuable business insights, making the project suitable for **academic submission, portfolios, and real‑world retail analytics use cases**.

---

## 👨‍💻 Author

* **Name:** Rami Reddy
* **Role:** Student / Data Science Enthusiast

---

## 📜 License

This project is licensed under the MIT License.
