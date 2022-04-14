
# Credit Risk Analysis
Predicting credit risk with Machine Learning models built and evaluated with Python

![Riks](https://user-images.githubusercontent.com/30667001/163343687-c8d26e19-66d9-40da-9b66-735c59abda88.png)

## Resources
- Data Source: LoanStats_2019Q1.csv
- Software: Python 3.7.6, Jupyter Notebook 6.4.5 
- Libraries: Pandas, NumPy 1.20.3, SciPy 1.7.1, Scikit-Learn 0.24.0

## Background
In an effort to improve the credit card application process and reduce the number individuals defaulting on loan repayment, Fast Lending - a peer-to-peer lending services company - applied Machine Learning models to predict the risk of extending credit to applicants. The goal was to maximize the identification of good candidates. However, as good candidates outweigh bad, additional sampling and modeling techniques were employed to improve the ability of the model to predict which individuals could be considered a potential risk for failing to repay the loan.

### Classification Model - General Steps
        # Train the model
        model.fit(X_train, y_train)
        # Evaluate model performance
        y_pred = model.predict(X_test)
        accuracy_score(y_test, y_pred)
        # Create a confusion matrix
        confusion_matrix(y_test, y_pred)
        # Generate a classification report
        classification_report(y_test, y_pred)

### Logistic Regression Model
![Logit2](https://user-images.githubusercontent.com/30667001/162936618-9c49a89a-cb0e-48c1-a607-6786512d4c7c.png)

Metrics used to validate model performance may be influenced by the number of events.
* Accuracy: Ratio of true positive and negative observations to all observations (TP + TN / n) </br>
  The Logit Model suggests an accuracy of 99.5% in its ability to predict good loan candidates over bad. As the number of good candidates vastly outweigh bad, the accuracy score overestimates the model's predictive performance due to overfitting. </br>

The classification report generates additional metrics for assessing model performance:
* Precision: Ratio of true positives to total predicted positives (TP / TP + FP) </br>
  This score is useful where missing a false positive could be costly (minimizes FP).</br>
  
* Recall: Ratio of true positives to total acutal positives (TP / TP + FN) </br>
  This score is useful when missing a false negative could be costly (minimizes FN).

Precision and recall can come at the expense of one another. An F1 score accommodates instances where both are important.
* F1 Score: A single score that provides a balance between recall and precision in a single value (2 * precision * recall / precision + recall)</br>

### Techniques to Address Overfitting of the Model
With metrics for evaluating a model determined, additional models were run to address the uneven number of good candidates versus bad.
1. Logistic Regression Models with Alternative Sampling to Address Class Imbalance </br>
   A. Naïve Random Oversampling </br>
   B. Synthetic Minority Oversampling Technique (SMOTE) </br>
   C. Cluster Centroid Undersampling </br>
   D. Combination Over-/Undersampling of Edited Nearest Neighbor and SMOTE (SMOTEENN)
2. Assembling Combination Models </br>
   A. Balanced Random Forest </br>
   B. Easy Ensemble Classification

## Results
![Predicting](https://user-images.githubusercontent.com/30667001/162927741-2d63d54a-f549-44b1-8fd7-30dfe83a851c.png)

### Metrics for All Models
| Model    | Accuracy | Precision | Recall     | F1 Score |
|----------|----------|-----------|------------|----------|
| ROS      | 0.65     | 0.99      | 0.65       | 0.79     |
| SMOTE    | 0.64     | 0.99      | 0.64       | 0.78     |
| Cluster  | 0.44     | 0.99      | 0.44       | 0.60     |
| SMOTEENN | 0.59     | 0.99      | 0.59       | 0.74     |
| RF       | 0.91     | 0.99      | 0.91       | 0.95     |
| EEC      | 0.94     | 0.99      | 0.94       | 0.97     |

## Summary

* Accurary - 
* Precision - 
* Recall - 
* F1 Score - 