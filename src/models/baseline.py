import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

from src.data.load_data import read_params

if __name__ == "__main__":
    # load in config params
    config = read_params(config_path="params.yaml")
    data_path = config["raw_data_config"]["raw_data_fold_csv"]
    target = config["raw_data_config"]["target"]
    # read training data
    df = pd.read_csv(data_path)

    # target mapping
    target_mapping = {"no": 0, "yes": 1}
    df.loc[:, target] = df[target].map(target_mapping)

    # total day charge has highest corr. with target
    feature = "total_day_charge"
    # loop over each fold
    n_splits = config["raw_data_config"]["n_splits"]
    scores_auc = []
    score_f1 = []
    for fold in range(n_splits):
        # seperate train and valid fold
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # target and features
        x_train = df_train[feature].values.reshape(-1, 1)
        y_train = df_train[target].values
        x_valid = df_valid[feature].values.reshape(-1, 1)
        y_valid = df_valid[target].values
        # train baseline model to output most common target
        clf_lr = LogisticRegression()
        clf_lr.fit(x_train, y_train)
        # predict on validation data and compute roc auc score
        y_preds = clf_lr.predict(x_valid)
        auc = roc_auc_score(y_valid, y_preds)
        f1 = f1_score(y_valid, y_preds)
        scores_auc.append(auc)
        score_f1.append(f1)
        # print auc score
        print(f"Fold = {fold}, AUC = {auc}")
        print(f"Fold = {fold}, F1 = {f1}")

    # average all auc
    average_auc = sum(scores_auc) / len(scores_auc)
    average_f1 = sum(score_f1) / len(score_f1)
    print(f"Average AUC = {average_auc}")
    print(f"Average F1 = {average_f1}")

    # mlflow
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("mlflow/churn_model")

    with mlflow.start_run(run_name="baseline model_lg") as mlops_run:
        mlflow.log_metric("auc", average_auc)
        mlflow.log_metric("f1", average_f1)
        # mlflow.log_notes("Single feature baseline logistic regression model")
