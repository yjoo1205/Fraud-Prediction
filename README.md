# Fraud Detection Models 
There are cases of frauds hidden among the legitimate transactions. These needs to be identified and mitigated to maintain trust and safeguard against financial deceit and misconduct. 
Data science can be tapped on to unveil these patterns, anomalies and irregularities. By leveraging advanced analytics and various modelling techniques, we can detect, mitigate risks and thereby prevent financial losses. 

## Table of Contents

- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Techniques Used](#techniques-used)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)

## Project Overview

This project aims to predict the fraudulent cases in a financial institution. 

## Datasets

Data used is Credit Card Fraud Detection data taken from Kaggle. 
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Transactions made by credit cards in September 2013 by European cardholders in two days.
- There are 492 frauds out of 284,807 transactions. 
- The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
  
It contains only numerical input variables which are the result of a PCA transformation. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. 

'Time' : contains the seconds elapsed between each transaction and the first transaction in the dataset. 
'Amount' : is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. 
'Class' : is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC) was recommended. Confusion matrix accuracy is not meaningful for unbalanced classification.

##  Techniques Used 
Machine Learning Models were used for this prediction. 
### 1. Random Forest Classifier 
#### Features and Advantages:
Random Forest Classifier is an ensemble learning method used for classification tasks.
It constructs multiple decision trees during training, each trained on a random subset of the training data and features.
The diversity among trees reduces overfitting compared to individual trees.
Predictions are then aggregated through majority voting or averaging.
This allows it to be robust against overfitting and noise in data.
Also, it is capable of capturing complex relationships and patterns.
#### Why is it useful for fraud detection?:
This is widely used across various domains, including fraud detection.
It is because it efficiently detects fraudulent activities by learning from diverse and complex data patterns.
Random Forest Classifier is a versatile and effective tool for classification tasks, particularly suitable for fraud detection applications due to its ability to handle complex data patterns and provide insights into feature importance. 

### 2. Gradient Boosting Machines (GBM)
#### Features and Advantages:
Gradient Boosting Machines (GBM) is a powerful ensemble learning technique used for classification and regression tasks.
It builds a strong predictive model by sequentially adding weak learners, typically decision trees, to the ensemble.
Unlike Random Forest, GBM builds trees sequentially, with each tree aiming to correct the errors made by the previous ones.
This iterative process allows GBM to focus more on difficult-to-predict instances, improving predictive accuracy.
GBM is particularly effective at capturing complex relationships and non-linear patterns in the data.
It handles both numerical and categorical features well and automatically selects the most informative features during training.
Additionally, GBM provides flexibility in hyperparameter tuning, allowing practitioners to optimize model performance for specific tasks.
#### Why is it useful for fraud detection?:
GBM is widely used across various domains, including fraud detection.
It efficiently detects fraudulent activities by learning from diverse and complex data patterns.
It is also a flexible and powerful tool for classification tasks, especially when dealing with imbalanced datasets and complex fraud patterns. Its robustness, high predictive power, and interpretability make it well-suited for fraud detection applications.

### 3. Support Vector Machines (SVM)
#### Features and Advantages:
Support Vector Machines (SVM) is a versatile supervised learning algorithm used for classification and regression tasks.
SVM works by finding the hyperplane that best separates the classes in the feature space while maximizing the margin between the classes.
This hyperplane is determined by support vectors, which are the closest data points to the decision boundary.
Also, SVM can handle high-dimensional data effectively and is robust to overfitting, especially in cases with limited training data.
Lastly, it can utilize different kernel functions, such as linear, polynomial, and radial basis function (RBF) kernels, to capture complex relationships between features.
#### Why is it useful for fraud detection?:
SVM is widely used across various domains, including fraud detection.
It efficiently separates fraudulent transactions from legitimate ones, even in highly imbalanced datasets.
Support Vector Machines offer a flexible and powerful tool for fraud detection, particularly in scenarios where linear separation is not sufficient and complex decision boundaries are needed. Its robustness, scalability, and ability to handle high-dimensional data make it well-suited for fraud detection applications.

### Model Performance Comparisons: 
Following methods were used to compare the performance of the models. 
- The classification report provides detailed information about precision, recall, F1-score, and other metrics for each class, helping to understand how well the model performs for different classes. 
- The confusion matrix gives a concise summary of correct and incorrect predictions, allowing for the analysis of specific types of errors made by the model. 
- Accuracy provides an overall measure of correct predictions, but it may not be sufficient, especially in the case of imbalanced datasets.
- Additionally, the AUC-ROC Score evaluates the model's ability to discriminate between positive and negative classes across different thresholds, providing valuable insights into its performance. 

Considering multiple evaluation metrics provides a comprehensive understanding of the model's performance, aiding in informed decision-making and model refinement.

## Limitations and Future Work

### Future Work
Feature Enhancement: Explore additional features and transformations to improve model performance.
Advanced Modeling: Investigate advanced modeling techniques such as deep learning and ensemble methods.
Imbalanced Data Handling: Develop strategies to address imbalance in the dataset for better model generalization.
Real-time Detection: Extend models for real-time detection to enable proactive fraud prevention.
Model Interpretability: Enhance model interpretability to facilitate understanding and trust among stakeholders.
Continuous Evaluation: Establish a framework for continuous model evaluation and updating to maintain effectiveness over time.

### Limitations
Imbalanced Data: The dataset may suffer from class imbalance, potentially leading to biased model predictions.
Feature Limitations: Limited availability or quality of features may impact the model's ability to capture fraudulent patterns accurately.
Generalization: Models may not generalize well to unseen data or new fraud patterns, requiring ongoing refinement and adaptation.

### References
https://www.linkedin.com/pulse/implementing-fraud-detection-algorithm-inpython-luis-soares-m-sc-
https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56
https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
