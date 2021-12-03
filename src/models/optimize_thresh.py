import pandas as pd
import numpy as np
import mlflow 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE

from src.data.load_data import read_params
from src.models.logistic_regression import feature_pipeline, score
from src.models.train_full import train_full

def optimize_threshold(clf, df, recall = 0.70):
    '''
    Uses cross-validation to optimize prob. threshold for required task
    :param clf: model to optimize
    :param df: training data with kfold col
    :param recall: desired recall 
    :return: optimal prob. threshold
    '''
    threshs = []
    dt_params = {'min_samples_split': 6, 'min_samples_leaf': 4, 'max_depth': 5, 'criterion': 'gini'}
    # loop over each fold
    for fold in range(5):
        # define train and validation set
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # target and features
        y_train = df_train['churn'].values
        y_valid = df_valid['churn'].values
        # fit model on training data 
        clf, features = train_full(df=df_train, model=DecisionTreeClassifier, params=dt_params)
        # get scores for valid data
        x_valid = features.transform(df_valid)
        y_scores = clf.predict_proba(x_valid)[:, 1]
        # locate where recall is closest to 0.70
        precisions, recalls, thresholds = precision_recall_curve(y_valid, y_scores)
        distance_to_optim = abs(recalls - recall)
        optimal_idx = np.argmin(distance_to_optim)
        threshs.append(thresholds[optimal_idx])
    # average optimal thresh across all folds
    return np.average(threshs)

if __name__ == '__main__':
    # load config
    config = read_params("params.yaml")
    data_path = config["raw_data_config"]["raw_data_fold_csv"]
    target = config["raw_data_config"]["target"]
    # read data
    df = pd.read_csv(data_path)
    # target mapping
    target_mapping = {"no": 0, "yes": 1}
    df.loc[:, target] = df[target].map(target_mapping)

    # optim threshold
    dt_params = {'min_samples_split': 6, 'min_samples_leaf': 4, 'max_depth': 5, 'criterion': 'gini'}
    clf, features = train_full(df=df, model=DecisionTreeClassifier, params=dt_params)
    thresh = optimize_threshold(clf, df)
    print(thresh)