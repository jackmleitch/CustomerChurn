# Customer Churn Prediction
## Project Overview
<!-- * Created a tool that **predicts kudos** (a proxy for user interaction) on [Strava activities](https://www.strava.com/athletes/5028644) (RMSE: 9.41) to see if it was random or if different attributes impact kudos in different ways.
* Attained over 4000 Strava activities using the **Strava API** and python.
* **Engineered new features** using domain knowledge. For example, features encapsulating different run types and times of day were added.
* Performed **feature selection** using a combination of SHAP values and feature importance.
* Optimized Linear (Lasso), Random Forest, and XGBoost Regressors using **Optuna** to reach the best model.
* Built an **interactive API** application using Streamlit, which can be found [here](https://strava-kudos.herokuapp.com/). -->

## Motivation


## Code and Resources Used 
**Python Version:** 3.9.7 <br />
**Packages:** pandas, numpy, sklearn, imblearn, mlxtend, mflow, shap, optuna, pytest, seaborn <br />
**Requirements:** ```pip install -r requirements.txt```  

## Project Write-Up
<!-- A blog post was written about this project and it was featured on Towards Data Science's editors pick section, it can be found [here](https://towardsdatascience.com/predicting-strava-kudos-1a4ce7a02053). -->

## Data Collection


## EDA
<!-- Some notable findings include:
* The Kudos received depended heavily on how many followers I had at the time. Unfortunately, there was no way to see how many followers I had at each point in time, therefore I could only use my most recent 1125 activities to train my model as the kudos stayed fairly consistent in this interval.
* It was found that the target variable is skewed right and there are some extreme values above ~100.
* Features such as distance, moving_time, and average_speed_mpk seem to share a similar distribution to the one we have with kudos_count.
* By looking at time distribution between activities, it was found that runs that are quickly followed in succession by other runs tend to receive fewer kudos than runs that were the only activity that day. To add to this, the longest activity of the day tends to receive more kudos than the other runs that day.

<p float="left">
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/pivot_table.png" width="200" />
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/indexs.png" width="300" /> 
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/corr.png" width="300" />
</p> -->

## Preprocessing and Feature Engineering
<!-- After obtaining the data, I needed to clean it up so that it was usable for my model. I made the following changes and created the following variables:
* An 80/20 train/test split was used and as my data contained dates, the most recent 20% of the data became the test set. I then split the training set into 5 folds using Sklearn's StratifiedKFold, Sturge's rule was used to bin the continuous target variable.
* Any missing values in a categorical feature were assigned a new category 'NONE' and missing values in numeric features were imputed using the median. Some heuristic functions were also used to impute systematic missing values. 
* Time-based features were added: year, month, day of the week, etc. Other features were also created using specific domain knowledge. I go into depth about this in the corresponding [blog post](https://towardsdatascience.com/predicting-strava-kudos-1a4ce7a02053).
* One-hot-encoding was used to encode categorical features and ordinal encoding was used to encode ordinal features. Target encoding was also used for a few categorical features.  -->

## Model Building 
<!-- In this step, I built a few different candidate models and compared different metrics to determine which was the best model for deployment. Three of those models were:
* Dummy Classifier (simply returns average kudos) - Baseline for the model.
* Lasso Regression - Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
* Random Forest - Again, with the sparsity associated with the data, I thought that this would be a good fit.
* XGBRegressor - Well... this model just always seems to work.

Feature selection was performed using a mix of SHAP values and feature importance from XGB. 

Optuna was used to tune all three shortlisted models. In particular, the Tree-structured Parzen Estimator (TPE) was used.
<p align="center">
<img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/optuna.png" width="600" height="300">
</p> -->

## Model Performance
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
</p> -->
