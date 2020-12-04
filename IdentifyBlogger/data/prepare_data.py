import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from IdentifyBlogger.data.BloggerDataset import BloggerDataset
from IdentifyBlogger.data.PhraseEncoder import PhraseEncoder


def encode_data(data: pd.DataFrame, category_map: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """
    encodes categorical data alongside provided map
    :param data: data to encode
    :param category_map: labels  encodeings
    :return: encoded data
    """
    for cat in category_map.keys():
        data[cat] = data[cat].map(category_map[cat])
    return data


def load_json(json_path: str) -> Dict:
    """
    loads json file from specified path
    :param json_path: path to json file
    :return: json file's content
    """
    with open(json_path, "r") as f:
        d = json.load(f)
        f.close()
    return d


def save_json(d: Dict, json_path: str) -> None:
    """
    saves object to json file in specified path
    :param d: object to save
    :param json_path: path to save object to
    """
    with open(json_path, "w") as f:
        json.dump(d, f)
        f.close()


def save_info(split: Dict[str, List[int]], category_map: Dict[str, Dict[str, int]], age_scaling_params: Dict[str, int],
              info_dir: str) -> None:
    """
    saves preprocesing info to specified directory
    :param split: data split
    :param category_map: category encoding map
    :param age_scaling_params: parameters for age scaling
    :param info_dir: path to saving directory
    """
    save_json(split, os.path.join(info_dir, "data_split.json"))
    save_json(category_map, os.path.join(info_dir, "category_map.json"))
    save_json(age_scaling_params, os.path.join(info_dir, "age_scaling_params.json"))


def load_info(info_dir: str) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    loads preprocesing info from specified directory
    :param info_dir: path to directory with preprocessing info
    :return: split, category_map, age_scaling_params
    """
    split = load_json(os.path.join(info_dir, "data_split.json"))
    category_map = load_json(os.path.join(info_dir, "category_map.json"))
    age_scaling_params = load_json(os.path.join(info_dir, "age_scaling_params.json"))
    return split, category_map, age_scaling_params


def split_data(data: pd.DataFrame, test_size: float = 0.1) -> Dict[str, List[int]]:
    """
    splits data  into train, validation and test subsets
    :param data: data to split
    :param test_size: relative sizes of test and validation datasets
    :return: split information
    """
    ids = data["id"].unique()
    np.random.shuffle(ids)
    test_bloggers = ids[:int(test_size * ids.size)]
    validation_bloggers = ids[test_bloggers.size:test_bloggers.size*2]
    train_bloggers = ids[validation_bloggers.size:]
    return {"train": train_bloggers.tolist(), "validation": validation_bloggers.tolist(), "test": test_bloggers.tolist()}


def generate_category_map(data: pd.DataFrame, categorical_vars: List[str]) -> Dict[str, Dict[str, int]]:
    """
    generates maps for encoding categorical variables
    :param data: data with variables to map
    :param categorical_vars: names of categorical variables
    :return: maps for encoding categorical variables
    """
    label_maps = {}
    for cat in categorical_vars:
        values = sorted(list(data[cat].unique()))
        label_maps[cat] = {v: i for i, v in enumerate(values)}
    return label_maps


def generate_info(data: pd.DataFrame, test_size: float = 0.1, categorical_variables: List[str] = None,
                  min_vocab_freq: int = 25) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, int]], Dict[str, int],
                                                     Dict[str, int]]:
    """
    generates data split, maps for encoding categorical variables, parameters for age scaling and vocabulary
    :param data: data
    :param test_size: relative sizes of test and validation datasets
    :param categorical_variables: names of categorical variables
    :param min_vocab_freq: minimum frequency in corpus for word to be included in vocabulary
    :return: split, category_map, age_scaling_params, vocabulary
    """
    if categorical_variables is None:
        categorical_variables = ["gender", "topic", "sign"]
    split = split_data(data, test_size=test_size)
    category_map = generate_category_map(data, categorical_vars=categorical_variables)
    train_ages = data["age"][data["id"].isin(split["train"])]
    age_scaling_params = {"min": train_ages.min(),
                          "max": train_ages.max()}
    enc = PhraseEncoder()
    vocabulary = enc.get_vocabulary(data["text"][data["id"].isin(split["train"])], min_freq=min_vocab_freq)

    return split, category_map, age_scaling_params, vocabulary


def preprocess_targets(data: pd.DataFrame, category_map: Dict[str, Dict[str, int]] = None,
                       age_scaling_params: Dict[str, int] = None) -> pd.DataFrame:
    """
    preprocesses target variables in data
    :param data: data to preprocess
    :param category_map: category encodings
    :param age_scaling_params: age scaling parameters
    :return: data with preprocessed targets
    """
    if "date" in data.columns:
        data = data.drop(["date"], axis=1)
    data = encode_data(data, category_map=category_map)
    data["age"] = (data["age"] - age_scaling_params["min"]) / age_scaling_params["max"]

    return data


def prepare_train_data(data: pd.DataFrame, validate: bool = True, preprocess_info_dir: str = None,
                       test_size: float = 0.1, categorical_variables: List[str] = None, min_vocab_freq: int = 25,
                       save_preprocess_info_dir: str = None) -> Tuple[BloggerDataset, BloggerDataset, Dict]:
    """
    prepares train and validation datasets for training and returns sme values required for building an nn
    :param data: data
    :param validate: whether to create validation dataset or not, if not validation data will be used in training and
    returned validation dataset will be None
    :param preprocess_info_dir: path to directory with preprocessing information; inf not None, preprocessing
    information will be loeded from this  directory
    :param test_size: relative sizes of test and validation datasets (inactive if loading preprocessing info)
    :param categorical_variables:  list of names of categorical variables
    :param min_vocab_freq: minimum frequency in corpus for word to be included in vocabulary (inactive if loading
    preprocessing info)
    :param save_preprocess_info_dir: defines whether preprocess info will be saved or not
    :return: train_dataset, validation_dataset, model_sizes
    """
    labels_order = ["gender", "age", "topic", "sign"]
    enc = PhraseEncoder()
    if preprocess_info_dir:
        split, category_map, age_scaling_params = load_info(preprocess_info_dir)
        enc.load_vocabulary(preprocess_info_dir)
    else:
        split, category_map, age_scaling_params, vocabulary = generate_info(data, test_size, categorical_variables,
                                                                            min_vocab_freq)
        enc.vocabulary = vocabulary

    if save_preprocess_info_dir:
        save_info(split, category_map, age_scaling_params, save_preprocess_info_dir)
        enc.save_vocabulary(save_preprocess_info_dir)

    if validate:
        train_data = data[data["id"].isin(split["train"])].drop(["id"], axis=1)
        validation_data = data.copy()[data["id"].isin(split["validation"])].drop(["id"], axis=1)
    else:
        train_data = data[data["id"].isin(split["train"] + split["validation"])].drop(["id"], axis=1)
        validation_data = None

    train_data = preprocess_targets(train_data, category_map, age_scaling_params)
    validation_data = None if validation_data is None else preprocess_targets(validation_data, category_map,
                                                                              age_scaling_params)

    train_data["text"] = train_data["text"].apply(enc.encode_phrase)
    train_data["text_length"] = train_data["text"].apply(len)
    train_data = train_data[train_data["text_length"] > 0]
    if validate:
        validation_data["text"] = validation_data["text"].apply(enc.encode_phrase)
        validation_data["text_length"] = validation_data["text"].apply(len)
        validation_data = validation_data[validation_data["text_length"] > 0]

    train_dataset = BloggerDataset(train_data)
    validation_dataset = None if validation_data is None else BloggerDataset(validation_data)

    output_dims = [1 if label in ["age", "gender"] else len(category_map[label]) for label in labels_order]
    model_sizes = {"output_dims": output_dims,
                   "vocab_size": enc.vocab_size,
                   "n_classes": {c: m for c, m in zip(labels_order, output_dims)}}

    return train_dataset, validation_dataset, model_sizes


def prepare_evaluation_data(data: pd.DataFrame, preprocess_info_dir: str = None) -> BloggerDataset:
    """
    prepares test dataset for network evaluation
    :param data: data
    :param preprocess_info_dir: path to directory with preprocessing information
    :return: test dataset
    """
    enc = PhraseEncoder()
    split, category_map, age_scaling_params = load_info(preprocess_info_dir)
    enc.load_vocabulary(preprocess_info_dir)
    test_data = data[data["id"].isin(split["test"])].drop(["id"], axis=1)
    test_data = preprocess_targets(test_data, category_map, age_scaling_params)
    test_data["text"] = test_data["text"].apply(enc.encode_phrase)
    test_data["text_length"] = test_data["text"].apply(len)
    test_data = test_data[test_data["text_length"] > 0]
    return BloggerDataset(test_data)
