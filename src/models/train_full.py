import pandas as pd
import numpy as np
import mlflow 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

from src.data.load_data import read_params
from src.models.logistic_regression import feature_pipeline, score

def train_full(df, model, params):
    """
    :param df: pandas dataframe containing our data
    :param model: model to train data on
    :param params: model hyperparameters
    :return: f1, recall, precision validation score for fold, as well as y_valid and preds
    """
    # feature pipeline
    features = feature_pipeline()

    # target and features
    x_train = features.fit_transform(df)
    y_train = df['churn'].values

    smt = SMOTE(random_state=42)
    x_train, y_train = smt.fit_resample(x_train, y_train)

    # create and train model
    clf = model(**params)
    clf.fit(x_train, y_train)
    return clf, features

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
    
    # train full model
    dt_params = {'min_samples_split': 6, 'min_samples_leaf': 4, 'max_depth': 5, 'criterion': 'gini'}
    clf, features = train_full(df=df, model=DecisionTreeClassifier, params=dt_params)