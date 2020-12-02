from typing import Dict, Union

import numpy as np
import pandas as pd


def encode_row(row: Dict, category_map: Dict[str, Dict[str, int]]) -> Dict:
    raise NotImplemented


def decode_row(row: Dict, category_map: Dict[str, Dict[str, int]]) -> Dict:
    raise NotImplemented


def encode_data(data: pd.DataFrame, category_map: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """

    :param data:
    :param category_map:
    :return:
    """
    assert category_map.keys() in data.columns, "encoded categories and column names don't match"
    for cat in category_map.keys():
        data[cat] = data[cat].apply(lambda row: category_map[cat][row[cat]], axis=1)
    return data


def decode_data(data: pd.DataFrame, category_map: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """

    :param data:
    :param category_map:
    :return:
    """
    enc2label = {cat: {v: k for k, v in category_map[cat].items()} for cat in category_map.keys()}
    for cat in enc2label.keys():
        data[cat] = data[cat].apply(lambda row: enc2label[cat][row[cat]], axis=1)
    return data


def rescale_column(column: pd.Series, minimum: float, maximum: Union[int, float]) -> pd.Series:
    """

    :param column:
    :param minimum:
    :param maximum:
    :return:
    """
