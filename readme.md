# 🚨 Ethereum Fraud Detection

This project uses **Machine Learning with XGBoost** to detect fraudulent transactions on the Ethereum blockchain. It includes full data preprocessing, class balancing with SMOTE, model training, hyperparameter tuning, and explainability through feature importance.

## 📂 Project Structure

ethereum-fraud-detection/
│
├── notebooks/
│ └── ethereum_fraud_detection.ipynb # Main notebook
├── models/
│ └── xgboost_fraud_model.pkl # Trained model
├── data/
│ ├── transactions.csv # Original dataset (if allowed)
│ ├── test_data.pkl # Test set
│ └── balanced_train_data.pkl # Resampled training set
├── README.md # Project description
├── requirements.txt # Required packages


## 🛠️ Technologies

- Python 3.x
- XGBoost
- scikit-learn
- imbalanced-learn (SMOTE)
- pandas, matplotlib, seaborn
- Google Colab

## 📊 Final Results (Optimized Model)

- **Accuracy**: 97%
- **Recall (Fraud class)**: 91%
- **Precision (Fraud class)**: 94%
- **F1-Score (Fraud class)**: 92%

## 🔍 Highlights

- Data cleaning and missing value handling
- SMOTE for balancing fraud vs non-fraud
- Hyperparameter tuning via `RandomizedSearchCV`
- Fraud detection using `XGBClassifier`
- Feature importance visualization

## 📁 Dataset

- Source: [Kaggle - Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)

## 🧠 Author

Developed by Cesar Ronai.

For more Data Science and Machine Learning projects, follow me on [GitHub](https://github.com/Cesar_ronai).

