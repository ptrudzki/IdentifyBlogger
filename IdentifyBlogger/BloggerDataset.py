from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


class BloggerDataset(Dataset):

    def __init__(self, data: pd.DataFrame) -> None:
        data["text_length"] = data["text"].apply(len, axis=1)
        self.data = data.applymap(torch.tensor)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        return self.data.iloc[item].to_dict()
