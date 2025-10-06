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

![Methodological Framework](https://github.com/Abhijeet-Santhosh/Predicting-Construction-Project-Costs-Using-Machine-Learning/blob/main/RF%20modelling%20methodology.png)

### 🔹 Data Source  
Data was sourced from **Kaggle (2025)** containing 14,507 projects registered under the **Gujarat Real Estate Regulatory Authority (RERA)** between 2017–2023. The dataset includes 44 attributes on promoter details, project timelines, costs, and size metrics, providing an extensive foundation for modelling.

### 🔹 Tools and Libraries  
Python (v3.11) with libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `shap`.  
Model development and visualisation were executed in **Jupyter Notebook** for full reproducibility.

### 🔹 Machine Learning Algorithm: Random Forest  
Random Forest (RF) is an ensemble learning algorithm that builds multiple decision trees and averages their outputs to enhance predictive accuracy and stability. It handles nonlinear relationships effectively and reduces overfitting through bootstrapping and feature randomisation.  

![Random Forest Algorithm](https://github.com/Abhijeet-Santhosh/Predicting-Construction-Project-Costs-Using-Machine-Learning/blob/main/RF%20framework.png) 

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
| **R² Score** | 1 - (Σ(yi - ŷi)² / Σ(yi - ȳ)²) | Variance explained by the model |
| **RMSE** | √(Σ(ŷi - yi)² / n) | Penalises large prediction errors |
| **MAE** | Σ|ŷi - yi| / n | Average deviation of predictions |

The Final RF Model achieved:  
- **R² = 0.9518**, **RMSE = 0.19**, **MAE = 0.13** (log scale)  
- After back-transforming: **RMSE ≈ ₹17.5M**, **MAE ≈ ₹11.5M**, proving strong predictive accuracy and stability.

---

## 6️⃣ Model Development and Results  

Hyperparameter tuning used **Out-of-Bag (OOB)** error analysis and **5-fold cross-validation** to identify optimal parameters:  

- n_estimators = 220
- max_features = 0.4
- max_depth = 15

Baseline comparisons showed Random Forest outperforming all models:  

![Model Performance Graph](https://github.com/Abhijeet-Santhosh/Predicting-Construction-Project-Costs-Using-Machine-Learning/blob/main/Model%20Performance%20Comparison.png)

---

## 7️⃣ Model Validation and Hyperparameter Tuning  

To ensure generalisability, **5-fold cross-validation** confirmed high stability with mean R² = 0.9503 (σ = 0.005).  
The **Out-of-Bag (OOB)** error convergence validated the ideal number of trees (n_estimators = 220).  

Together, these tests confirmed low overfitting risk and strong model reliability across varying data partitions.

---

## 8️⃣ Final RF Model Feature Analysis  

Feature interpretation combined three complementary methods:  

1. **Feature Importance** (reduction in impurity)
![Feature Importance Graph](https://github.com/Abhijeet-Santhosh/Predicting-Construction-Project-Costs-Using-Machine-Learning/blob/main/Feature%20Importance%20Graph.png)  
2. **Sensitivity Analysis** (impact magnitude on predictions)
![Sensitivity Analysis](https://github.com/Abhijeet-Santhosh/Predicting-Construction-Project-Costs-Using-Machine-Learning/blob/main/Sensitivity%20Analysis.png)
3. **SHAP Values** (game-theoretic explanation of individual predictions)  
![SHAP Summary Plot](https://github.com/Abhijeet-Santhosh/Predicting-Construction-Project-Costs-Using-Machine-Learning/blob/main/SHAP%20analysis.png)

The top-ranked features were:
1. **Promoter Cost Group**  
2. **Carpet Area (log)**  
3. **Engineer Cost Group**  
4. **Architect Cost Group**  
5. **Land Cost (log)**  

These variables had the strongest influence on project cost predictions.

---

## 9️⃣ Final RF Model Cost Drivers and Industry Validation  

The consolidated **Final Feature Score** was derived as:  
Final Feature Score_i = 0.39·FI_i + 0.08·Sensitivity_i + 0.53·SHAP_i

| Rank | Feature | Final Score | Description |
|:----:|:---------|:-------------|:-------------|
| 1 | Promoter Cost Group | 0.66 | Developer scale & reputation strongly impact total cost. |
| 2 | Carpet Area (log) | 0.27 | Larger project sizes lead to exponential cost increases. |
| 3 | Engineer Cost Group | 0.18 | Structural and MEP complexity influence cost variance. |
| 4 | Architect Cost Group | 0.17 | Design quality and expertise directly affect total cost. |
| 5 | Land Cost (log) | 0.14 | Location-based acquisition costs are major budget components. |

Industry comparison validated these drivers as consistent with studies (Gleeds, 2022; Parmar et al., 2016), reinforcing the model’s realism and predictive robustness.  

---

## 🔟 Ethical Consideration  

This study exclusively used **public, anonymised secondary data** from the Gujarat RERA database, retrieved via Kaggle (2025). No primary or human-participant data were collected.  

Ethical approval was granted through the **University of Exeter Business School’s ethical framework**, ensuring compliance with the principles of integrity, confidentiality, and transparency. Data was securely stored on **OneDrive (University account)**, version-controlled, and documented for reproducibility.  

All analysis adhered to **UEBS ethical guidelines** and **British Psychological Society (BPS) principles**, ensuring responsible data use, accurate reporting, and avoidance of bias or misuse of findings.

---

### 📚 Citation  
**Abhijeet Santhosh Kumar (2025).**  
*Regional Cost Estimation Model of Real Estate Projects Using Random Forest: A Case Study of Gujarat RERA Data.*  
University of Exeter, MSc Business Analytics Dissertation.
