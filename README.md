💳 Credit Card Fraud Detection using Bagging – ANN Project

This project focuses on detecting fraudulent credit card transactions using Artificial Neural Networks (ANN) with "Bagging (Bootstrap Aggregation)". Built as part of an academic assignment on ensemble learning techniques.

Objectives
- Handle highly imbalanced data from the Kaggle Credit Card Fraud dataset
- Apply "bagging ensemble" strategy with neural networks as base models
- Evaluate model using metrics focused on minority class detection
- Visualize performance with confusion matrix and ROC curve

ML & DL Techniques
- Artificial Neural Networks (ANN)  
- Bagging (ensemble learning)  
- Class balancing (e.g., undersampling/SMOTE)

Tools & Libraries
- Python  
- Keras / TensorFlow  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn

Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions with only 492 fraud cases (~0.17%)
- Features anonymized (V1 to V28), plus 'Time' and 'Amount'

Evaluation Metrics
- Accuracy (limited use in imbalanced data)  
- Precision, Recall, F1-score  
- Confusion Matrix  
- ROC Curve and AUC Score
