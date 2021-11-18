import os
import argparse
import pandas as pd
from src.data.load_data import read_params
from sklearn.model_selection import StratifiedKFold


def stratKFold(config_path):
    """
    Perform stratified K fold cross validation on training set
    :param df: pd dataframe to split
    :param target: target variable 
    :param n_splits: number of folds 
    :return: df with kfold column
    """

    # read in config params
    config = read_params(config_path)
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    target = config["raw_data_config"]["target"]
    n_splits = config["raw_data_config"]["n_splits"]
    random_state = config["raw_data_config"]["random_state"]

    df = pd.read_csv(raw_data_path)
    # create new column 'kfold' with val -1
    df["kfold"] = -1
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # target values
    y = df[target].values
    # initialise kfold class
    kf = StratifiedKFold(n_splits=n_splits)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    return df


if __name__ == "__main__":
    # read in params.yaml path from command line (if given)
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    df_split = stratKFold(config_path=parsed_args.config)
    config = read_params(parsed_args.config)
    df_split.to_csv(config["raw_data_config"]["raw_data_kfold_csv"], index=False)

