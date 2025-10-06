# 🏗️ Regional Cost Estimation Model of Real Estate Projects Using Random Forest  
**A Case Study of Gujarat RERA Data**

---

## 1️⃣ Project Overview  

India’s construction and real estate sector contributes nearly 50% of the nation’s GDP (directly and indirectly) and continues to grow rapidly with urbanisation and investment inflows. Yet, chronic cost overruns remain a persistent issue—438 monitored infrastructure projects exceeded budgets by ₹5.18 lakh crore (≈61.8%), largely due to inaccurate early-stage estimates. Traditional linear regression methods fail to capture the complex, nonlinear cost interactions that occur in real projects.  

This project develops a **machine learning-based prediction model** to enhance the accuracy of cost estimation in the Indian real estate sector. Using **Gujarat RERA (GujRERA)** data, the study builds a **Random Forest regression model** that learns from 14,507 project records. It applies data-driven techniques to uncover dominant cost drivers while benchmarking model performance against conventional regression methods.  

### 🧩 Research Gap  
Despite RERA providing a rich and standardised dataset, few studies have applied ML for predictive cost estimation in India. This project fills that gap by integrating **machine learning, feature engineering, and interpretability tools (SHAP and sensitivity analysis)** within the RERA framework, offering one of the first regional cost estimation models in the Indian context.  

---

## 2️⃣ Aim and Objectives  

**Aim:**  
To develop a machine learning-based regional cost estimation model that predicts project costs using Gujarat RERA data, improving early-stage budget accuracy and transparency.  

**Objectives:**  
- Perform descriptive and exploratory data analysis on Gujarat’s real estate market using GujRERA data.  
- Build and optimise a **Random Forest regression model** for project cost prediction.  
- Compare its performance against baseline models: **Linear Regression**, **Polynomial Regression**, and **Support Vector Regression (SVR)**.  
- Identify and validate the **key cost drivers** using feature importance, sensitivity, and SHAP analyses, supported by industry evidence.  

---

## 3️⃣ Methodology  

The research follows the **CRISP-DM framework**, integrating data collection, preprocessing, model development, validation, and interpretation. The Random Forest model was chosen for its robustness in handling nonlinearity and multicollinearity in high-dimensional construction datasets.

### 🔹 Data Source  
Data was sourced from **Kaggle (2025)** containing 14,507 projects registered under the **Gujarat Real Estate Regulatory Authority (RERA)** between 2017–2023. The dataset includes 44 attributes on promoter details, project timelines, costs, and size metrics, providing an extensive foundation for modelling.

### 🔹 Tools and Libraries  
Python (v3.11) with libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `shap`.  
Model development and visualisation were executed in **Jupyter Notebook** for full reproducibility.

### 🔹 Machine Learning Algorithm: Random Forest  
Random Forest (RF) is an ensemble learning algorithm that builds multiple decision trees and averages their outputs to enhance predictive accuracy and stability. It handles nonlinear relationships effectively and reduces overfitting through bootstrapping and feature randomisation.  

![Random Forest Algorithm](/images/random_forest_algorithm.png)  
![Methodological Framework](/images/methodology_framework.png)  

---

## 4️⃣ Data Cleaning and Processing  

The preprocessing workflow involved:  
- **Missing value imputation** using median/mode techniques.  
- **Outlier treatment** and **log transformations** to normalise skewed variables such as `totalEstimatedCost`, `totalLandCost`, and `totalCarpetArea_form3A`.  
- **Multicollinearity reduction** through **Pearson Correlation Coefficient (PCC)** and **Variance Inflation Factor (VIF)** tests, removing redundant variables.  
- **Feature engineering:** creation of `projectDurationMonths` and **cost-based binning** of high-cardinality features like `promoterName`, `architect_name`, and `eng_name`.  
- **Label encoding** of categorical variables for Random Forest compatibility.  

These steps improved model interpretability, reduced distortion, and ensured a robust dataset for predictive modelling.

---

## 5️⃣ Model Performance Metrics  

Model performance was assessed using standard regression metrics:  

| Metric | Formula | Interpretation |
|:-------|:---------|:---------------|
| **R² Score** - Variance explained by the model |
| **RMSE** - Penalises large prediction errors |
| **MAE** - Average deviation of predictions |

The Final RF Model achieved:  
- **R² = 0.9518**, **RMSE = 0.19**, **MAE = 0.13** (log scale)  
- After back-transforming: **RMSE ≈ ₹17.5M**, **MAE ≈ ₹11.5M**, proving strong predictive accuracy and stability.

---

## 6️⃣ Model Development and Results  

Hyperparameter tuning used **Out-of-Bag (OOB)** error analysis and **5-fold cross-validation** to identify optimal parameters:  
