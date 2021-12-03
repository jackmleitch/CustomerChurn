import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from src.data.load_data import read_params
from src.models.logistic_regression import feature_pipeline, score
from src.models.train_full import train_full

def predict(X, clf, feature_pipeline, thresh=0.71):
    '''
    Predict on new data 
    :param X: data containing features
    :param clf: trained model
    :param feature_pipeline: trained feature processing pipeline
    :param thresh: prediction threshold
    :return: predictions
    '''
    X = feature_pipeline.transform(X)
    preds = (clf.predict_proba(X)[:,1] >= thresh).astype(int)
    return preds

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

    # load in test data and score
    df_test = pd.read_csv("data/external/test.csv")
    # target mapping
    target_mapping = {"no": 0, "yes": 1}
    df_test.loc[:, 'churn'] = df_test['churn'].map(target_mapping)
    y_test = df_test['churn'].values
    # predict on unseen data
    preds = predict(df_test, clf, features)
    # score
    f1, recall, precision = score(y_test, preds)
    print(f"Average F1 = {f1}, Recall = {recall}, Precision = {precision}")