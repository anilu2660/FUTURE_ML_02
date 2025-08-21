# 📊 Customer Churn Prediction System

This project was developed during my **Machine Learning Internship at Future Interns**.  
The goal of this project is to build a machine learning model that predicts **whether a customer is likely to churn** (leave/cancel a service) based on their profile and usage behavior.  

---

## 🚀 Project Overview
Customer churn is a major challenge for businesses, especially in **telecom, SaaS, and subscription-based industries**.  
By predicting churn early, companies can take proactive measures to **retain customers** and reduce losses.  

In this project:
- The dataset is based on **Telco Customer Churn data**.  
- Features include **contract type, payment method, tenure, monthly charges, total charges**, and various service options.  
- Target variable: **Churn** (`Yes` / `No` → encoded as `1` / `0`).  

---

## 🛠️ Tech Stack & Tools
- **Python** 🐍  
- **Pandas / NumPy** → Data preprocessing  
- **Matplotlib / Seaborn** → Data visualization  
- **Scikit-learn** → Encoding, scaling, evaluation metrics  
- **XGBoost** → Machine learning model  

---

## 📂 Dataset
- **Source**: Telco Customer Churn Dataset (7,043 rows × 21 columns)  
- Key Columns:
  - `customerID` → Unique ID (dropped)  
  - `tenure` → Number of months customer stayed  
  - `MonthlyCharges` → Current monthly bill  
  - `TotalCharges` → Total amount charged  
  - `Contract`, `PaymentMethod`, `InternetService`, etc.  

---

## ⚙️ Workflow
1. **Data Preprocessing**  
   - Dropped `customerID`  
   - Converted `TotalCharges` to numeric and handled missing values  
   - Encoded categorical features using `LabelEncoder`  
   - Scaled numerical features  

2. **Model Building**  
   - Used **XGBoost Classifier**  
   - Tuned parameters: `n_estimators`, `learning_rate`, `max_depth`, etc.  
   - Addressed class imbalance using `scale_pos_weight`  

3. **Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix  
   - Precision-Recall Curve  
   - Feature Importance Analysis  

---

## 📊 Results
- Baseline accuracy: **~81%**  
- Good performance in predicting **non-churn customers**  
- Scope for improvement in **recall for churn customers** using SMOTE & hyperparameter tuning  

---

## 📈 Visualizations
- Churn distribution plot  
- Feature correlation heatmap  
- Feature importance (XGBoost)  
- Confusion matrix  
- Precision-Recall curve  

---

## ✨ Learnings
- Hands-on experience with **real-world ML workflows**  
- Gained skills in **data preprocessing, feature engineering, model evaluation, and visualization**  
- Understood the business importance of **recall vs. accuracy** in churn prediction  

---

## 🙌 Acknowledgement
This project was completed as part of my **Machine Learning Internship at Future Interns**.  
A big thanks to **Future Interns** for the opportunity and guidance.  

---

## 🔖 Tags
`#MachineLearning` `#DataScience` `#ChurnPrediction` `#CustomerRetention` `#XGBoost` `#futureinterns`
