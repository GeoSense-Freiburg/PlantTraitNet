import pandas as pd
from typing import List, Dict, Tuple
from .logger import get_logger
from omegaconf import ListConfig

def get_train_columns(data: pd.DataFrame, target_prefix: List[str]) -> Dict[str, List[str]]:
    """
    Extract target-related columns (median, std, bounds) based on given prefixes.

    Args:
        data (pd.DataFrame): Input dataset.
        target_prefix (List[str]): Prefixes identifying target columns (e.g., ["X"]).

    Returns:
        Dict[str, List[str]]: Mapping of target column categories.
    """
    target_columns = [
        col for prefix in target_prefix for col in data.columns if col.startswith(prefix)
    ]

    target_median_columns = [c for c in target_columns if "_median" in c]
    target_std_columns = [c for c in target_columns if "_sd" in c]
    target_lowerbound_columns = [c for c in target_columns if "_lowerbound" in c]
    target_upperbound_columns = [c for c in target_columns if "_upperbound" in c]

    return {
        "target_columns": target_columns,
        "target_median_columns": target_median_columns,
        "target_std_columns": target_std_columns,
        "target_lowerbound_columns": target_lowerbound_columns,
        "target_upperbound_columns": target_upperbound_columns,
    }



def get_eval_columns(target_prefix) -> Dict[str, List[str]]:
    """
    Prepare column mapping for evaluation or inference.
    Ensures ListConfig (OmegaConf) is converted to a standard list.
    """
    if isinstance(target_prefix, ListConfig):
        target_prefix = list(target_prefix)
    return {"target_label_columns": target_prefix}


def inf_transform_target(
    test_data: pd.DataFrame,
    columns: Dict[str, List[str]],
    target_transformer,
) -> pd.DataFrame:
    """
    Transform target columns in the test dataset using a provided transformer.

    Args:
        test_data (pd.DataFrame): The test dataset.
        columns (dict): Dictionary containing 'target_label_columns'.
        target_transformer: Fitted transformer to apply to target columns.

    Returns:
        pd.DataFrame: Transformed test dataset.
    """
    logger = get_logger()

    if target_transformer is None:
        raise ValueError("target_transformer must be provided and cannot be None.")

    target_columns = columns["target_label_columns"]  
    logger.info(f"Target columns for transformation: {target_columns}")

    if not isinstance(target_columns, list):
        raise ValueError("'target_label_columns' must be a list of column names.")

    test_target_values = test_data[target_columns].values
    logger.info(f"Original target shape: {test_target_values.shape}")

    transformed_values = target_transformer.transform(test_target_values)
    logger.info(f"Transformed target shape: {transformed_values.shape}")

    if transformed_values.shape != test_target_values.shape:
        raise ValueError(
            f"Shape mismatch: original {test_target_values.shape}, transformed {transformed_values.shape}"
        )

    for idx, col in enumerate(target_columns):
        test_data[col] = transformed_values[:, idx]

    return test_data


def parse_dataset(
    meta_config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Parse training and validation datasets based on meta configuration.

    Args:
        meta_config (dict): Must contain 'path_imgref_train', 'path_imgref_val', and 'target_prefix'.

    Returns:
        Tuple: (train_data, val_data, columns, val_columns)
    """
    logger = get_logger()

    required_keys = ["path_imgref_train", "path_imgref_val", "target_prefix"]
    for key in required_keys:
        if key not in meta_config:
            raise KeyError(f"Missing required key '{key}' in meta_config.")

    train_data = pd.read_csv(meta_config["path_imgref_train"])
    val_data = pd.read_csv(meta_config["path_imgref_val"])

    columns = get_train_columns(train_data, meta_config["target_prefix"])
    val_columns = get_eval_columns(meta_config["target_prefix"])

    logger.debug(f"Training columns extracted: {columns.keys()}")

    # Note: No transformation applied here. Done inside Dataset class after label sampling.
    return train_data, val_data, columns, val_columns


def parse_inf_dataset(meta_config: dict, target_transformer) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Parse inference dataset (test or validation) and optionally apply target transformation.

    Args:
        meta_config (dict): Must contain 'target_prefix' and either 'path_imgref_test' or 'path_imgref_val'.
        target_transformer: Transformer used to scale/normalize targets.

    Returns:
        Tuple: (inference_data, columns)
    """
    logger = get_logger()

    if "target_prefix" not in meta_config:
        raise KeyError("'target_prefix' must be present in meta_config.")

    # Prefer test data if available, otherwise fallback to validation data
    if "path_imgref_test" in meta_config and meta_config["path_imgref_test"]:
        data_path = meta_config["path_imgref_test"]
        split_type = "test"
    elif "path_imgref_val" in meta_config and meta_config["path_imgref_val"]:
        data_path = meta_config["path_imgref_val"]
        split_type = "validation"
    else:
        raise KeyError("Either 'path_imgref_test' or 'path_imgref_val' must be present in meta_config.")

    logger.info(f"Loading {split_type} dataset from {data_path}")
    inf_data = pd.read_csv(data_path)
    columns = get_eval_columns(meta_config["target_prefix"])

    missing_cols = [c for c in columns["target_label_columns"] if c not in inf_data.columns]
    if missing_cols:
        logger.warning(f"Skipping target transformation. Missing columns: {missing_cols}")
    else:
        logger.info(f"Applying target transformation to {split_type} data.")
        inf_data = inf_transform_target(inf_data, columns, target_transformer)

    return inf_data, columns

