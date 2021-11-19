import pandas as pd
import mlflow
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

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

    # loop over each fold
    n_splits = config["raw_data_config"]["n_splits"]
    scores = []
    for fold in range(n_splits):
        # seperate train and valid fold
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # target and features
        x_train = df_train.drop(columns=target, axis=1).values
        y_train = df_train[target].values
        x_valid = df_valid.drop(columns=target, axis=1).values
        y_valid = df_valid[target].values
        # train baseline model to output most common target
        clf_dummy = DummyClassifier()
        clf_dummy.fit(x_train, y_train)
        # predict on validation data and compute roc auc score
        y_preds = clf_dummy.predict(x_valid)
        auc = roc_auc_score(y_valid, y_preds)
        scores.append(auc)
        # print auc score
        print(f"Fold = {fold}, AUC = {auc}")
    # average all auc
    average_auc = sum(scores) / len(scores)

    # mlflow
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("mlflow/churn_model")

    with mlflow.start_run(run_name="baseline model") as mlops_run:
        mlflow.log_metric("auc", average_auc)
