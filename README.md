# Customer Churn Prediction
## Project Overview
* Created an interpretable model that **predicts churn** on telecom data found on [Kaggle](https://www.kaggle.com/c/customer-churn-prediction-2020/overview) (F1: 0.796) to build a pipeline to try and retain more customers.
* Used **SMOTE** to oversample minority class (churned customers) which imporved the F1 score from 0.61 to 0.80.
* **Engineered new features** for example, aggregating different features together. 
* Performed **feature selection** using SHAP values.
* Optimized Logistic Regression and Decision Tree models using **Optuna** and pruned the decision tree further to reduce overfitting.
* Used Mlflow to track model experimentation. 

## Motivation
I've always believed that to truly learn data science you need to practice data science and I wanted to do this project to practice working with imbalanced classes in classification problems. This was also a perfect opportunity to start working with mlflow to help track my machine learning experiments: it allows me to track the different models I have used, the parameters I've trained with, and the metrics I've recorded.

This project was aimed at predicting customer churn using the telecommunications data found on [Kaggle](https://www.kaggle.com/c/customer-churn-prediction-2020/overview). That is, we want to be able to predict if a given customer is going the leave the telecom provider based on the information we have on that customer. Now, why is this useful? Well, if we can predict which customers we think are going to leave before they leave then we can try to do something about it! For example, we could target them with specific offers, and maybe we could even use the model to provide us insight into what to offer them because we will know, or at least have an idea, as to why they are leaving.

## Code and Resources Used 
**Python Version:** 3.9.7 <br />
**Packages:** pandas, numpy, sklearn, imblearn, mlxtend, mflow, shap, optuna, pytest, seaborn <br />
**Requirements:** ```pip install -r requirements.txt```  

## Project Write-Up
<!-- A blog post was written about this project and it was featured on Towards Data Science's editors pick section, it can be found [here](https://towardsdatascience.com/predicting-strava-kudos-1a4ce7a02053). -->

## Data Collection
The data for this project was from a public Kaggle competition found [here](https://www.kaggle.com/c/customer-churn-prediction-2020/overview). 

## EDA
Some notable findings include:
* The target variable was very imbalanced with 85.93% of the data belonging to class churn=no. 
* There were not many correlated features when feature interaction was not taken into account. 

<p float="left">
  <img src="https://github.com/jackmleitch/CustomerChurn/blob/main/data/blog_content/churn_plot.png" width="500" />
  <img src="https://github.com/jackmleitch/CustomerChurn/blob/main/data/blog_content/corr.png" width="400" /> 
</p>

## Preprocessing and Feature Engineering
After obtaining the data, I needed to clean it up so that it was usable for my model. I made the following changes and created the following variables:
* An stratified 80/20 train/test split was used. I then split the training set into 5 folds using Sklearn's StratifiedKFold.
* The yes/no target values were mapped to binary 1/0. 
* Any missing values in a categorical feature were assigned a new category 'NONE' and missing values in numeric features were imputed using the median. 
* Aggregation of some features was used, for example combining total day, eve, and night calls into one feature total calls.
* One-hot-encoding was used to encode categorical features and ordinal encoding was used to encode ordinal features. 

## Model Building 
In this step, I started by building two baseline models which were a simple Dummy Classifier which always predicts the majority class (churn=no) and a Logistic Regression model that used the two most correlated features with churn. After this, I built a few different candidate models and compared different metrics to determine which was the best model for deployment. My main aim was to build an interpretable model so the two shortlisted models were:
* Logistic Regression 
* Decision Tree Classifier

Feature selection was performed using SHAP values from a more complicated Random Forest Classifier. 
<p align="center">
<img src="https://github.com/jackmleitch/CustomerChurn/blob/main/data/blog_content/shap.png" width="400" height="300">
</p>

Optuna was used to tune both shortlisted models. In particular, the Tree-structured Parzen Estimator (TPE) was used to minimize a custom metric which was the RMSE of the difference between validation and training F1 score and is weighted four times less than the validation F1 score. Optimizing this function forced the train and valid score to stay close, while also maximizing the validation score. This was used as decision trees notoriously overfit the training data. Cost complexity post pruning was also used to help the decision tree generalize better (and also improve interpretability). 

<!-- ## Model Performance
<!-- The XGB model far outperformed the other approaches on the test and validation sets.
* XGB Regressor : RMSE = 9.41
* Lasso Regression: RMSE = 10.54
* Random Forest Regressor: RMSE = 10.89

The final XGB model was then trained on the whole training dataset using the hyperparameters found in the Optuna experiment. -->

## Model Interpretation
<!-- SHAP was used to interpret the model and individual predictions. It was found that longer and faster runs recieve more kudos (as they are more impressive?). Workouts also recieve more kudos than easy runs.

<p float="left">
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/shap_feature_imp.png" width="400" />
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/shap1.png" width="400" /> 
</p> --> -->
