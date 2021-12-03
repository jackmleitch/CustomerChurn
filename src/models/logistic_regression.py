import pandas as pd
import numpy as np
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import recall_score, f1_score, precision_score
from mlxtend.feature_selection import ColumnSelector
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

from src.data.load_data import read_params


# scoring function 
def score(y, preds):
    """
    Returns corresponding metric scores 
    :param y: true y values
    :param preds: predicted y values
    :return: f1_score, recall, and precision scores 
    """
    f1 = f1_score(y, preds)
    recall = recall_score(y, preds)
    precision = precision_score(y, preds)
    return [f1, recall, precision]


def feature_pipeline(config_path="params.yaml"):
    """
    :param config_path: path to params.yaml file
    :return: preprocessing feature pipeline 
    """
    # load in config information
    config = read_params(config_path)
    num_features = config["raw_data_config"]["model_features"]["numeric"]
    cat_features = config["raw_data_config"]["model_features"]["categorical"]
    # transformers
    transforms = []
    # categorical pipeline
    transforms.append(
        (
            "catergorical",
            Pipeline(
                [
                    ("select", ColumnSelector(cols=cat_features)),
                    ("encode", OneHotEncoder()),
                ]
            ),
        )
    )
    # numeric pipeline
    transforms.append(
        (
            "numeric",
            Pipeline(
                [
                    ("select", ColumnSelector(cols=num_features)),
                    ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
                    ("scale", StandardScaler()),
                ]
            ),
        )
    )

    # combine features
    features = FeatureUnion(transforms)
    return features


def run(fold, config_path="params.yaml", smote=False):
    """
    :param fold: fold to train model on
    :param config_path: path to params.yaml file
    :return: f1, recall, precision validation score for fold
    """
    # load in config information
    config = read_params(config_path)
    data_path = config["raw_data_config"]["raw_data_fold_csv"]
    target = config["raw_data_config"]["target"]

    # read data
    df = pd.read_csv(data_path)

    # target mapping
    target_mapping = {"no": 0, "yes": 1}
    df.loc[:, target] = df[target].map(target_mapping)

    features = feature_pipeline()

    # define train and validation set
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # target and features
    y_train = df_train[target].values
    y_valid = df_valid[target].values

    # create training and validation features
    x_train = features.fit_transform(df_train)
    x_valid = features.transform(df_valid)

    # smote x tomek
    if smote:
        # smt = SMOTETomek(
        #     smote=SMOTE(sampling_strategy=0.35, random_state=42, n_jobs=-1),
        #     tomek=TomekLinks(sampling_strategy="majority", n_jobs=-1),
        #     random_state=42,
        #     n_jobs=-1,
        # )
        smt = SMOTE(random_state=42)
        x_train, y_train = smt.fit_resample(x_train, y_train)

    # create and train model
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid)
    # score model
    f1, recall, precision = score(y_valid, preds)
    print(f"Fold = {fold}, F1 = {f1}")
    print(f"Fold = {fold}, Recall = {recall}")
    print(f"Fold = {fold}, Precision = {precision}")
    return f1, recall, precision


if __name__ == "__main__":
    config = read_params("params.yaml")
    folds = config["raw_data_config"]["n_splits"]
    scores_f1 = []
    scores_recall = []
    scores_precision = []
    for i in range(folds):
        f1, recall, precision = run(fold=i, smote=True)
        scores_f1.append(f1)
        scores_recall.append(recall)
        scores_precision.append(precision)

    average_f1 = sum(scores_f1) / len(scores_f1)
    average_recall = sum(scores_recall) / len(scores_recall)
    average_precision = sum(scores_precision) / len(scores_precision)
    print(f"\nAverage F1 = {average_f1}")
    print(f"\nAverage Recall = {average_recall}")
    print(f"\nAverage Precision = {average_precision}")

    # mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("mlflow/churn_model")

    with mlflow.start_run(run_name="linear_regression_smote") as mlops_run:
        mlflow.log_metric("F1", average_f1)
        mlflow.log_metric("Recall", average_recall)
        mlflow.log_metric("Precision", average_precision)
