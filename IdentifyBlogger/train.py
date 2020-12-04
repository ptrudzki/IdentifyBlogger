import argparse
import os
from typing import Tuple

import pandas as pd

from IdentifyBlogger.BloggerDataset import BloggerDataset
from IdentifyBlogger.IdentityLSTM import IdentityLSTM
from IdentifyBlogger.PhraseEncoder import PhraseEncoder
from IdentifyBlogger.preprocess_targets import preprocess_data
from IdentifyBlogger.train_loop import train_loop


def train(data_path: str, test_size: float = 0.1, validate: bool = True) -> None:

    labels_order = ["gender", "age", "topic", "sign"]

    print("Loading data")
    data = pd.read_csv(data_path)
    # data = pd.read_csv(data_path, index_col=0)
    data, split, category_map, age_scaling_params = preprocess_data(data, test_size=test_size)
    if validate:
        train_data = data[data["id"].isin(split["train"])].drop(["id"], axis=1)
        validation_data = data[data["id"].isin(split["validation"])].drop(["id"], axis=1)
    else:
        train_data = data[data["id"].isin(split["train"] + split["validation"])].drop(["id"], axis=1)
        validation_data = None

    # enc = PhraseEncoder(train_data["text"], min_freq=25)
    enc = PhraseEncoder()
    enc.load_vocabulary("./utils")
    print("encoding text")
    train_data["text"] = train_data["text"].apply(enc.encode_phrase)
    train_data["text_length"] = train_data["text"].apply(len)
    train_data = train_data[train_data["text_length"] > 0]
    if validate:
        validation_data["text"] = validation_data["text"].apply(enc.encode_phrase)
        validation_data["text_length"] = validation_data["text"].apply(len)
        validation_data = validation_data[validation_data["text_length"] > 0]


    # train_data = pd.read_csv("./temp_train.csv", index_col=0)
    # train_data = train_data["text"].apply(lambda x: [int(i) for i in x.replace(',', "").replace('[', "").replace(']', "").split()])[0]
    # validation_data = pd.read_csv("./temp_validation.csv", index_col=0)

    print("dupa")
    train_dataset = BloggerDataset(train_data)
    validation_dataset = None if validation_data is None else BloggerDataset(validation_data)

    output_dims = [1 if label in ["age", "gender"] else len(category_map[label]) for label in labels_order]
    # output_dims = [1]
    activations = ["Sigmoid", None, "Softmax", "Softmax"]
    criterions = ["BCELoss", "MSELoss", "CrossEntropyLoss", "CrossEntropyLoss"]
    # activations = ["Sigmoid"]
    # criterions = ["BCELoss"]

    model = IdentityLSTM(vocab_size=enc.vocab_size, output_dims=output_dims, activations=activations, n_layers=1)

    train_loop(model, train_dataset, validation_dataset, criterions=criterions, device="cuda", n_epochs=5,
               num_workers=0, batch_size=64)


if __name__ == '__main__':
    p = "D:/data/blogtext/blogtext.csv"
    # p = "./temp_data"
    train(p)