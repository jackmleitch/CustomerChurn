from src.data.split_data import stratKFold
from src.data.load_data import read_params


def test_stratKFold():
    """
    Testing stratified kfold function.
    """
    config = read_params("params.yaml")
    target = config["raw_data_config"]["target"]
    n_splits = config["raw_data_config"]["n_splits"]

    df = stratKFold(config_path="params.yaml")

    assert df.kfold.nunique() == n_splits
