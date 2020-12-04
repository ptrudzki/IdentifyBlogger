import json
import os
import re
from typing import Dict, List
import pandas as pd


class PhraseEncoder:
    """
    class for generating and storing vocabulary set and using it to encode phrases
    """

    def __init__(self) -> None:
        """
        initializing encoder as empty or using data from specified directory
        :param data_path: path to specified directory
        :param min_freq: minimum number of times word has to occur in dataset to be added to vocabulary
        """
        self.vocabulary = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def is_empty(self) -> bool:
        """
        encoder is empty if it's vocabulary is empty
        """
        return len(self.vocabulary) == 0

    def get_vocabulary(self, data: pd.Series, min_freq: int = 3) -> Dict[str, int]:
        """
        creates vocabulary, set of words occurring in train data subset with corresponding integer encoding
        :param data:
        :param min_freq: minimum number of times word has to occur in dataset to be added to vocabulary
        :return: vocabulary
        """
        vocabulary = {'xxpad': 0, 'xxunk': 1}
        freq_map = {}
        for text in data:
            text = re.sub('[^\w ]', ' ', text.lower())
            text = text if not text.startswith(' ') else text[1:]
            for token in text.split():
                if token not in freq_map.keys():
                    freq_map[token] = 0
                else:
                    freq_map[token] += 1
        for token, freq in freq_map.items():
            if freq >= min_freq:
                vocabulary[token] = len(vocabulary)
        return vocabulary

    def set_vocabulary(self, data: pd.Series, min_freq: int = 3) -> None:
        self.vocabulary = self.get_vocabulary(data, min_freq)

    def load_vocabulary(self, dirpath: str) -> None:
        """
        loads vocabulary from vocabulary.json file from specified directory
        :param model_dirpath: path to specified directory
        """
        with open(os.path.join(dirpath, "vocabulary.json"), "r") as f:
            self.vocabulary = json.load(f)
        f.close()

    def save_vocabulary(self, dirpath: str) -> None:
        """
        saves vocabulary to vocabulary.json file in specified directory
        :param model_dirpath: path to specified directory
        """
        with open(os.path.join(dirpath, "vocabulary.json"), "w") as f:
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
        phrase = re.sub('^\w ', ' ', phrase.lower())
        phrase = phrase if not phrase.startswith(' ') else phrase[1:]
        return [self._lookup_enc(token) for token in phrase.split()]

    def encode_text(self, text: pd.Series) -> pd.Series:
        """

        :param text:
        :return:
        """
        return text.apply(self.encode_phrase)
