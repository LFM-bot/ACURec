import copy
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from src.utils.utils import neg_sample


def load_specified_dataset(model_name, config):
    return SequentialDataset


class BaseSequentialDataset(Dataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(BaseSequentialDataset, self).__init__()
        self.batch_dict = {}
        self.num_items = config.num_items
        self.config = config
        self.train = train
        self.dataset = config.dataset
        self.max_len = config.max_len
        self.item_seq = data_pair[0]
        self.label = data_pair[1]

    def get_SRtask_input(self, idx):
        item_seq = self.item_seq[idx]
        target = self.label[idx]
        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        return (torch.tensor(item_seq, dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))

    def __getitem__(self, idx):
        return self.get_SRtask_input(idx)

    def __len__(self):
        return len(self.item_seq)

    def collate_fn(self, x):
        return self.basic_SR_collate_fn(x)

    def basic_SR_collate_fn(self, x):
        """
        x: [(seq_1, len_1, tar_1), ..., (seq_n, len_n, tar_n)]
        """
        item_seq, seq_len, target = default_collate(x)
        self.batch_dict['item_seq'] = item_seq
        self.batch_dict['seq_len'] = seq_len
        self.batch_dict['target'] = target
        return self.batch_dict


class SequentialDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(SequentialDataset, self).__init__(config, data_pair, additional_data_dict, train)


if __name__ == '__main__':
    index = np.arange(10)
    res = np.random.choice(index, size=1)
    print(index)
    print(res)
