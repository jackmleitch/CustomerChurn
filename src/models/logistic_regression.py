import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import ColumnSelector
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

from src.data.load_data import read_params


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
    :return: auc validation score for fold
    """
    # load in config information
    config = read_params(config_path)
    data_path = config["raw_data_config"]["raw_data_fold_csv"]
    target = config["raw_data_config"]["target"]
    num_features = config["raw_data_config"]["model_features"]["numeric"]
    cat_features = config["raw_data_config"]["model_features"]["categorical"]

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
    auc = roc_auc_score(y_valid, preds)
    print(f"Fold = {fold}, AUC = {auc}")
    return auc


if __name__ == "__main__":
    config = read_params("params.yaml")
    folds = config["raw_data_config"]["n_splits"]
    scores = []
    for i in range(folds):
        auc = run(fold=0, smote=True)
        scores.append(auc)
    print(f"\nAverage AUC = {sum(scores)/len(scores)}")
