import argparse
import os
from typing import Tuple

import pandas as pd

from IdentifyBlogger.IdentityLSTM import IdentityLSTM
from IdentifyBlogger.prepare_data import prepare_train_data
from IdentifyBlogger.train_loop import train_loop


def train(data_path: str, test_size: float = 0.1, validate: bool = True, preprocess_info_dir: str = None,
          min_vocab_freq: int = 25, save_preprocess_info_dir: str = None) -> None:
    # data = pd.read_csv(data_path)
    data = pd.read_csv(data_path, index_col=0)
    categorical_variables = ["gender", "topic", "sign"]
    train_dataset, validation_dataset, model_sizes = prepare_train_data(data, validate=validate,
                                                                        preprocess_info_dir=preprocess_info_dir,
                                                                        test_size=test_size,
                                                                        categorical_variables=categorical_variables,
                                                                        min_vocab_freq=min_vocab_freq,
                                                                        save_preprocess_info_dir=save_preprocess_info_dir)

    activations = ["Sigmoid", None, "Softmax", "Softmax"]
    criterions = ["BCELoss", "MSELoss", "CrossEntropyLoss", "CrossEntropyLoss"]

    model = IdentityLSTM(vocab_size=model_sizes["vocab_size"], output_dims=model_sizes["output_dims"],
                         activations=activations, n_layers=4, embedding_size=32, hidden_size=512)

    train_loop(model, train_dataset, validation_dataset, criterions=criterions, device="cuda", n_epochs=5,
               num_workers=0, batch_size=64)


if __name__ == '__main__':
    # p = "D:/data/blogtext/blogtext.csv"
    p = "./temp_data"
    train(p, min_vocab_freq=3)
