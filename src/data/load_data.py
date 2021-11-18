import yaml
import argparse
import numpy as np
import pandas as pd


def read_params(config_path):
    """
    Read parameters from the params.yaml file
    :param config_path: params.yaml location
    :return: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def load_data(data_path, model_features=False):
    """
    Load csv dataset from given path
    :param data_path: csv path to data
    :param model_var: attributed to load 
    :return: pandas dataframe 
    """
    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    if model_features:
        df = df[model_features]
    return df


def load_raw_data(config_path):
    """
    Load data from external location(data/external) to the raw folder(data/raw) with train and testing dataset 
    :param config_path: path to config file 
    :param: save train file in data/raw folder 
    """
    config = read_params(config_path)
    external_data_path = config["external_data_config"]["external_data_csv"]
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    df = load_data(external_data_path)
    df.to_csv(raw_data_path, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)
