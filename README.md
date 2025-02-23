# 🚀 Credit Card Default Prediction Project

## 🎯 Project Overview

Welcome to the Credit Card Default Prediction Project! This project is designed to predict whether a credit card holder is likely to default on their payments, helping financial institutions make informed decisions and mitigate risks.

We leveraged powerful machine learning techniques to create a robust predictive model, with XGBoost emerging as the superstar of our model lineup!

Here's the workflow of the model

![alt text](<Blank diagram.png>)

And here's the workflow of the User Interface

![alt text](<Blank diagram (1).png>)

## 💡 Why This Matters

Credit card defaults can lead to significant financial losses for banks and lenders. By predicting defaulters in advance, financial institutions can take proactive steps, like adjusting credit limits, reaching out to customers, or implementing tailored repayment plans.

This project is a step towards smarter, data-driven financial management!

## 📊 The Data

Our dataset is packed with information about credit card holders, including their payment history, demographic details, and billing amounts. Here’s a sneak peek at the key features:

### 🔢 Numerical Columns
- `LIMIT_BAL`: Credit limit of the cardholder.
- `AGE`: Age of the cardholder.
- `BILL_AMT_*`: Billing amounts from April to September.
- `PAY_AMT_*`: Payment amounts from April to September.
- `PAY_*`: Payment status for each month.

### 🧩 Categorical Columns
- `SEX`: Gender of the cardholder.
- `EDUCATION`: Education level of the cardholder.
- `MARRIAGE`: Marital status of the cardholder.

### 🎯 Target Column
- `Is_Defaulter`: 1 if the cardholder is likely to default, 0 otherwise.

## 🔧 Data Preprocessing

To ensure our model performs at its best, we took the following preprocessing steps:

- **Balanced the Data:** Used SMOTE to address the class imbalance and ensure our model isn’t biased.
- **Encoded Categorical Variables:** Applied one-hot encoding to transform categorical variables into a format the model can digest.
- **Standardized Numerical Data:** Scaled numerical columns to bring them onto a similar scale, enhancing model performance.

## 🧠 Model Training & Evaluation

We trained several models, and after rigorous testing, XGBoost took the crown as the best-performing model. Here’s how it scored:

### 🏆 Training Performance
- **Accuracy:** 98.90%
- **Precision:** 99.60%
- **Recall:** 98.30%
- **F1 Score:** 98.90%
- **AUC:** 98.90%

### 🔍 Validation Performance
- **Accuracy:** 86.60%
- **Precision:** 89.90%
- **Recall:** 82.30%
- **F1 Score:** 86.00%
- **AUC:** 86.60%

These results speak volumes about our model’s ability to accurately predict defaulters!

## 🚀 Getting Started

To get this project up and running on your local machine, clone the repository and install the required dependencies:

```bash
git clone https://github.com/rayyan666/creditcard-default-prediction-using-ML.git
cd creditcard-default-prediction-using-ML
pip install -r requirements.txt
python app.py
```

 ## Documentation

- [Low-Level Design (LLD) Document](https://drive.google.com/file/d/1-WNVGfdnh6XQcTIKv439RUmVEtRx50TR/view?usp=sharing)
- [High-Level Design (HLD) Document](https://drive.google.com/file/d/1fJ8R8qXCpjokonYyovBKzhmE73frIXJM/view?usp=sharing)
