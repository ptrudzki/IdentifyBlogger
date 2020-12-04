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
        # data["text_length"] = data["text"].apply(len)
        # self.data = data.applymap(torch.tensor)
        self.data = data.applymap(torch.tensor)
        # self.data = data.applymap(lambda x: torch.tensor(x, device="cuda"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        return self.data.iloc[item].to_dict()


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from IdentifyBlogger.preprocess_targets import preprocess_data
    from IdentifyBlogger.PhraseEncoder import PhraseEncoder
    import time
    DATA_PATH = "D:/data/blogtext/blogtext.csv"
    data = pd.read_csv(DATA_PATH)
    data, split, category_map, age_scaling_params = preprocess_data(data)
    start = time.time()
    enc = PhraseEncoder(data["text"], min_freq=25)
    enc.save_vocabulary("./utils")
    print(time.time() - start)
    # enc = PhraseEncoder()
    # enc.load_vocabulary("./utils")
    start = time.time()
    data["text"] = enc.encode_text(data['text'])
    print(time.time() - start)
    dataset = BloggerDataset(data)
    dl = DataLoader(dataset, collate_fn=collate, batch_size=5)
    x = next(iter(dl))
    print(len(data))
