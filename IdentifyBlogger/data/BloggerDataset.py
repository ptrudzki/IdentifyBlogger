from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


def collate(batch) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    function for creating batches from unequal length phrases
    :param batch: batch of data to concatenate
    :return: model ready batch
    """
    batch.sort(key=lambda x: x["text_length"], reverse=True)
    gender, age, topic, sign, encoded_text, lengths = zip(*[d.values() for d in batch])
    encoded_text = torch.nn.utils.rnn.pad_sequence(encoded_text, batch_first=True)
    gender = torch.stack(gender)
    age = torch.stack(age)
    sign = torch.stack(sign)
    topic = torch.stack(topic)
    lengths = torch.stack(lengths)
    return encoded_text, lengths, {"gender": gender.float(), "age": age.float(), "topic": topic, "sign": sign}


class BloggerDataset(Dataset):

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data.applymap(torch.tensor)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        return self.data.iloc[item].to_dict()
