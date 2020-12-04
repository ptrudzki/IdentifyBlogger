import argparse
import os
from typing import Tuple, Union

import pandas as pd
import torch

from IdentifyBlogger.IdentityLSTM import IdentityLSTM
from IdentifyBlogger.prepare_data import prepare_train_data
from IdentifyBlogger.train_loop import train_loop


def train(data_path: str, test_size: float = 0.1, validate: bool = True, preprocess_info_dir: str = None,
          min_vocab_freq: int = 25, save_preprocess_info_dir: str = None, n_layers=4, embedding_size=32,
          hidden_size: int = 512, n_epochs: int = 5, num_workers: int = 0, batch_size: int = 64,
          device: Union[str, torch.device] = "cpu") -> None:
    """
    function to train blog author identity model
    :param data_path: path to data csv file
    :param test_size: size of test dataset in relative to whole dataset (inactive if loading preprocessing info)
    :param validate: defines whether validation will be performed or not
    :param preprocess_info_dir: path to directory with previously saved preprocessing info to load; if None,
    new preprocessing info will be generated
    :param min_vocab_freq: minimum frequency in corpus for word to be included in vocabulary (inactive if loading preprocessing info)
    :param save_preprocess_info_dir: defines whether preprocess info will be saved or not
    :param n_layers: number of lstm layers
    :param embedding_size: size of embeddings in embedding layers
    :param hidden_size: size of lstm's hidden layer
    :param n_epochs: number of epochs to train for
    :param num_workers: number of data loading workers
    :param batch_size: number of data entries loaded in one batch
    :param device: device to train on
    """
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
                         activations=activations, n_layers=n_layers, embedding_size=embedding_size,
                         hidden_size=hidden_size)

    train_loop(model, train_dataset, validation_dataset, criterions=criterions, device=device, n_epochs=n_epochs,
               num_workers=num_workers, batch_size=batch_size)


if __name__ == '__main__':
    # p = "D:/data/blogtext/blogtext.csv"
    p = "./temp_data"
    train(p, min_vocab_freq=3)
