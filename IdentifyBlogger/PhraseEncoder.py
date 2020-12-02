import json
import os
from typing import Tuple, Dict, List
import pandas as pd
import spacy


class PhraseEncoder:
    """
    class for generating and storing vocabulary set and using it to encode phrases
    """
    nlp = spacy.load('en_core_web_sm')

    def __init__(self, data_path: str = None, min_freq: int = 3) -> None:
        """
        initializing encoder as empty or using data from specified directory
        :param data_path: path to specified directory
        :param min_freq: minimum number of times word has to occur in dataset to be added to vocabulary
        """
        if data_path is None:
            self.vocabulary = {}
        else:
            self.vocabulary = self._get_vocabulary(data_path, min_freq)

    @property
    def is_empty(self) -> bool:
        """
        encoder is empty if it's vocabulary is empty
        """
        return len(self.vocabulary) == 0

    def _get_vocabulary(self, data_dir: str, min_freq: int = 3) -> Dict[str, int]:
        """
        creates vocabulary, set of words occurring in train data subset with corresponding integer encoding
        :param data_dir: path to directory with data
        :param min_freq: minimum number of times word has to occur in dataset to be added to vocabulary
        :return: vocabulary
        """
        raise NotImplemented

    def load_vocabulary(self, model_dirpath: str) -> None:
        """
        loads vocabulary from vocabulary.json file from specified directory
        :param model_dirpath: path to specified directory
        """
        with open(os.path.join(model_dirpath, "vocabulary.json"), "r") as f:
            self.vocabulary = json.load(f)
        f.close()

    def save_vocabulary(self, model_dirpath: str) -> None:
        """
        saves vocabulary to vocabulary.json file in specified directory
        :param model_dirpath: path to specified directory
        """
        with open(os.path.join(model_dirpath, "vocabulary.json"), "w") as f:
            json.dump(self.vocabulary, f)
        f.close()

    def _lookup_enc(self, token: str) -> int:
        """
        returns token id in vocabulary or unknown token id if token is not in vocabulary
        :param token:
        :return:
        """
        return self.vocabulary["xxunk"] if token not in self.vocabulary.keys() else self.vocabulary[token]

    def encode_phrase(self, phrase: str) -> List[int]:
        """
        tokenizes and encodes passed phrase
        :param phrase:
        :return:
        """
        return [self._lookup_enc(token.text) for token in self.nlp(phrase.lower(),
                                                                   disable=['parser', 'tagger', 'ner'])]

