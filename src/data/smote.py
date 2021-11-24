import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

from src.models.logistic_regression import feature_pipeline
from src.data.load_data import read_params

if __name__ == "__main__":
    config = read_params("params.yaml")
    data_path = config["raw_data_config"]["raw_data_fold_csv"]
    target = config["raw_data_config"]["target"]

    # read data
    df = pd.read_csv(data_path)

    # target mapping
    target_mapping = {"no": 0, "yes": 1}
    df.loc[:, target] = df[target].map(target_mapping)

    # process data
    features = feature_pipeline()

    # smote x tomek
    smt = SMOTETomek(
        smote=SMOTE(sampling_strategy=0.35, random_state=42, n_jobs=-1),
        tomek=TomekLinks(sampling_strategy="majority", n_jobs=-1),
        random_state=42,
        n_jobs=-1,
    )

    # training data
    y_train = df[target].values
    x_train = features.fit_transform(df)

    X_smt, y_smt = smt.fit_resample(x_train, y_train)
