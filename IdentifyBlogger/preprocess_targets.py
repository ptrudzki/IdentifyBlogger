import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from IdentifyBlogger.utils import encode_data


def split_data(data: pd.DataFrame, test_size: float = 0.1) -> Dict[str, List[int]]:
    """

    :param test_size:
    :param data:
    :return:
    """
    ids = data["id"].unique()
    np.random.shuffle(ids)
    test_bloggers = ids[:int(test_size * ids.size)]
    validation_bloggers = ids[test_bloggers.size:test_bloggers.size*2]
    train_bloggers = ids[validation_bloggers.size:]
    return {"train": train_bloggers, "validation": validation_bloggers, "test": test_bloggers}


def generate_category_map(data: pd.DataFrame, categorical_vars: List[str]) -> Dict[str, Dict[str, int]]:
    """

    :param categorical_vars:
    :param data:
    :return:
    """
    label_maps = {}
    for cat in categorical_vars:
        values = sorted(list(data[cat].unique()))
        label_maps[cat] = {v: i for i, v in enumerate(values)}
    return label_maps


def preprocess_data(data: pd.DataFrame, test_size: float = 0.1, categorical_variables: List[str] = None,
                    split_dir: str = None, category_map_dir: str = None,
                    age_scaling_params_path: Dict[str, int] = None) -> Tuple[pd.DataFrame, Dict[str, List[int]],
                                                                             Dict[str, Dict[str, int]], Dict[str, int]]:
    """

    :param data:
    :param test_size:
    :param categorical_variables:
    :param split_dir:
    :param category_map_dir:
    :param age_scaling_params_path:
    :return:
    """
    if categorical_variables is None:
        categorical_variables = ["gender", "topic", "sign"]

    data = data.drop(["date"], axis=1)

    if split_dir is None:
        split = split_data(data, test_size=test_size)
    else:
        with open(os.path.join(split_dir, "data_split.json"), "r") as f:
            split = json.load(f)
            f.close()

    if category_map_dir is None:
        category_map = generate_category_map(data, categorical_vars=categorical_variables)
    else:
        with open(os.path.join(split_dir, "category_map.json"), "r") as f:
            category_map = json.load(f)
            f.close()

    if age_scaling_params_path is None:
        train_ages = data["age"][data["id"].isin(split["train"])]
        age_scaling_params = {"min": train_ages.min(),
                              "max": train_ages.max()}
    else:
        with open(os.path.join(split_dir, "age_scaling_params.json"), "r") as f:
            age_scaling_params = json.load(f)
            f.close()

    data = encode_data(data, category_map=category_map)
    data["age"] = (data["age"] - age_scaling_params["min"]) / age_scaling_params["max"]

    return data, split, category_map, age_scaling_params
