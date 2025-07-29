
<h1 align="center">💳 Credit Card Fraud Detection using Logistic Regression</h1>

<p align="center">
A binary classification machine learning project to detect fraudulent credit card transactions using transaction features and logistic regression. Built with Python and Scikit-learn.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

---

## 🧠 Problem Statement

Given anonymized credit card transaction data, the goal is to build a model that classifies whether a transaction is legitimate or fraudulent.
This model can assist financial institutions in minimizing financial loss due to fraud.

---

## 📊 Dataset Details

* **Name:** Credit Card Fraud Detection Dataset
* **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Instances:** 284,807 transactions
* **Features:**

  * `Time`: Seconds elapsed between this and the first transaction
  * `V1` to `V28`: Principal Components from PCA (anonymized)
  * `Amount`: Transaction amount
  * `Class`: Target variable

    * `0` = Legitimate transaction
    * `1` = Fraudulent transaction

---

## 🚀 Workflow Overview

1. **Import Libraries** – Load essential Python packages
2. **Load & Explore Data** – Read and understand data distribution
3. **Handle Imbalance** – Apply under-sampling for balanced learning
4. **Feature & Target Split** – Separate inputs and output variable
5. **Train-Test Split** – Stratified sampling to maintain class ratio
6. **Train Logistic Regression** – Fit model on training data
7. **Evaluate Model** – Accuracy, confusion matrix, classification report

---

## 🧪 Why Logistic Regression?

* Simple yet effective for binary classification problems
* Fast training and interpretable results
* Works well on linearly separable data
* Useful as a baseline before applying complex models

---

## 📈 Model Accuracy

| Dataset  | Accuracy |
| -------- | -------- |
| Training | \~94.4%  |
| Testing  | \~94.4%  |

*Additional metrics such as precision, recall, and F1-score were used due to data imbalance.*

---

## 📦 Installation & Setup

```bash
git clone https://github.com/Toshaksha/credit-card-fraud-detector.git
cd credit-card-fraud-detector
pip install -r requirements.txt
```

---

## 💡 Sample Confusion Matrix & Metrics

```
Confusion Matrix:
[[95  4]
 [ 7 91]]

Classification Report:
              precision    recall  f1-score
       Legit     0.93       0.96      0.95
       Fraud     0.96       0.93      0.94
```

---

## 📝 How to Use

* Open `credit_card_fraud_detection.ipynb` in Jupyter Notebook or Google Colab.
* Run the notebook to train the model, view results, and test predictions.
* You can modify or try out new sample transactions to check predictions.

---

## 📂 Files & Structure

```
credit-card-fraud-detection/
│
├── data                         
│   └── [Download manually from Kaggle]   # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
│
├── credit_card_fraud_detection.ipynb     # Main Jupyter Notebook
├── requirements.txt                      # Dependencies
├── .gitignore                            # Git ignore file
└── README.md                             # Project documentation
```

---

## 🧰 Tech Stack

* Python 3.x
* NumPy
* Pandas
* Scikit-learn

---

## 👤 Author

**Toshaksha** – [GitHub Profile](https://github.com/Toshaksha)

Let’s connect and collaborate on more projects!

---

⭐ **If this project helped you, feel free to give it a star!**

