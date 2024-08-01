import copy
import math
import random
import numpy as np
import torch
import math
import torch.nn.functional as F


class AbstractDataAugmentor:
    def __init__(self, aug_ratio):
        self.aug_ratio = aug_ratio

    def transform(self, item_seq, seq_len):
        """
        :param item_seq: torch.LongTensor, [batch, max_len]
        :param seq_len: torch.LongTensor, [batch]
        :return: aug_seq: torch.LongTensor, [batch, max_len]
        """
        raise NotImplementedError


class CauseDrop(AbstractDataAugmentor):
    """
    Torch version of item drop operation.
    """

    def __init__(self, drop_ratio):
        super(CauseDrop, self).__init__(drop_ratio)
        self.drop_ratio = drop_ratio

    def set_ratio(self, drop_ratio):
        self.drop_ratio = 1 - drop_ratio

    def transform(self, seq, score, p=None):
        """
        Parameters
        ----------
        seq: [batch_size, max_len]
        score: [batch_size, max_len]

        Returns
        -------
        aug_item_seq: [batch_size, max_len]
        aug_seq_len: [batch_size]
        """
        dev = seq.device
        B, L = seq.shape
        p = self.drop_ratio if p is None else p
        k = int(L * p)

        # sample drop item indices
        drop_indices = torch.multinomial(score, num_samples=k)  # [B, drop_size]

        # set sampled items as 0
        tmp_seq = seq.scatter(-1, drop_indices, 0).long()

        # mask matrices for valid item
        valid_item_mask = (tmp_seq > 0).bool()
        valid_size = valid_item_mask.sum(-1).unsqueeze(-1)  # [B]

        # mask matrices for new positions
        position_tensor = torch.arange(L).repeat(B, 1).to(dev)  # [B, L]
        valid_pos_mask = (position_tensor < valid_size).bool()

        # move valid items to new positions
        dropped_item_seq = torch.zeros_like(seq).to(dev)
        dropped_item_seq[valid_pos_mask] = tmp_seq[valid_item_mask]

        # post-processing: avoid all 0 item
        invalid_mask = (valid_size == 0).bool()
        invalid_mask = invalid_mask.repeat(1, L)
        invalid_mask[:, 1:] = 0
        dropped_item_seq[invalid_mask] = seq[invalid_mask]
        valid_size = (dropped_item_seq > 0).sum(-1)  # [B]

        return dropped_item_seq, valid_size
