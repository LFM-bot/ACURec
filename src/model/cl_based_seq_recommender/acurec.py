import random
import sys
import torch.nn.functional as F
from src.model.abstract_recommeder import AbstractRecommender
import argparse
import torch
import torch.nn as nn
from src.model.sequential_encoder import Transformer
from src.model.loss import InfoNCEWithExtraNeg
from src.utils.utils import HyperParamDict
from src.model.data_augmentation import *


class ACURec(AbstractRecommender):
    def __init__(self, config, additional_data_dict):
        super(ACURec, self).__init__(config)
        self.mask_id = self.num_items
        self.embed_size = config.embed_size
        self.initializer_range = config.initializer_range
        self.tao = config.tao
        self.lamda = config.lamda

        self.item_embedding = nn.Embedding(self.num_items, self.embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, self.embed_size)
        self.input_layer_norm = nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
        self.input_dropout = nn.Dropout(config.hidden_dropout)
        self.trm_encoder = Transformer(embed_size=self.embed_size,
                                       ffn_hidden=config.ffn_hidden,
                                       num_blocks=config.num_blocks,
                                       num_heads=config.num_heads,
                                       attn_dropout=config.attn_dropout,
                                       hidden_dropout=config.hidden_dropout,
                                       layer_norm_eps=config.layer_norm_eps)
        self.pos_aug = CauseDrop(drop_ratio=1 - config.essential_ratio)
        self.neg_aug = CauseDrop(drop_ratio=config.essential_ratio)
        self.nce_with_neg = InfoNCEWithExtraNeg(temperature=self.tao,
                                                similarity_type='dot')
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def train_forward(self, data_dict):
        item_seq, seq_len, target = self.load_basic_SR_data(data_dict)
        seq_embedding, all_attn_scores = self.seq_encoding(item_seq, seq_len,
                                                           return_attn_score=True)
        candidates = self.item_embedding.weight
        logits = seq_embedding @ candidates.t()
        rec_loss = self.cross_entropy(logits, target)

        # cl task
        cl_loss = self.calc_cl_loss(item_seq, seq_len, target, seq_embedding, all_attn_scores)

        return rec_loss + self.lamda * cl_loss

    def forward(self, data_dict):
        item_seq, seq_len, _ = self.load_basic_SR_data(data_dict)
        seq_embedding = self.seq_encoding(item_seq, seq_len)
        candidates = self.item_embedding.weight
        logits = seq_embedding @ candidates.t()

        return logits

    def position_encoding(self, item_input):
        seq_embedding = self.item_embedding(item_input)
        position = torch.arange(self.max_len, device=item_input.device).unsqueeze(0)
        position = position.expand_as(item_input).long()
        pos_embedding = self.position_embedding(position)
        seq_embedding += pos_embedding
        seq_embedding = self.input_layer_norm(seq_embedding)
        seq_embedding = self.input_dropout(seq_embedding)

        return seq_embedding

    def seq_encoding(self, item_seq, seq_len, return_all=False, return_attn_score=False):
        seq_embedding = self.position_encoding(item_seq)
        out_seq_embedding, all_attn_score = self.trm_encoder(item_seq, seq_embedding,
                                                             out_attn_score=True)
        if not return_all:
            out_seq_embedding = self.gather_index(out_seq_embedding, seq_len - 1)
        if not return_attn_score:
            return out_seq_embedding
        return out_seq_embedding, all_attn_score

    def calc_cl_loss(self, item_seq, seq_len, target, seq_embeddings, all_attn_scores):
        # aggregate attention score
        attn_scores = torch.stack(all_attn_scores, 0).mean(0)  # [B, L, L]
        attn_score = attn_scores[:, -1, :].squeeze()  # [B, L]
        # attn_score = self.gather_index(attn_scores, seq_len - 1)

        # construct pos & neg views
        seq_pos, len_pos = self.pos_aug.transform(item_seq, 1 - attn_score)
        seq_neg, len_neg = self.neg_aug.transform(item_seq, attn_score)

        # view encoding
        seq_embs_pos = self.seq_encoding(seq_pos, len_pos)
        seq_embs_neg = self.seq_encoding(seq_neg, len_neg)

        # InfoNCE with extra neg samples
        view_view_cl_loss = self.nce_with_neg(seq_embeddings, seq_embs_pos, extra_neg=seq_embs_neg)

        return view_view_cl_loss


def ACURec_config():
    parser = HyperParamDict('ACURec default hyper-parameters')
    parser.add_argument('--model', default='ACURec', type=str)
    parser.add_argument('--essential_ratio', default=0.3, type=float)
    parser.add_argument('--model_type', default='Sequential', choices=['Sequential', 'Knowledge'])
    parser.add_argument('--tao', default=1., type=float, help='temperature for softmax')
    parser.add_argument('--lamda', default=0.1, type=float,
                        help='weight for contrast learning loss, only work when jointly training')
    # Transformer
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--ffn_hidden', default=512, type=int, help='hidden dim for feed forward network')
    parser.add_argument('--num_blocks', default=2, type=int, help='number of transformer block')
    parser.add_argument('--num_heads', default=2, type=int, help='number of head for multi-head attention')
    parser.add_argument('--hidden_dropout', default=0.5, type=float, help='hidden state dropout rate')
    parser.add_argument('--attn_dropout', default=0.5, type=float, help='dropout rate for attention')
    parser.add_argument('--layer_norm_eps', default=1e-12, type=float, help='transformer layer norm eps')
    parser.add_argument('--initializer_range', default=0.02, type=float, help='transformer params initialize range')

    parser.add_argument('--loss_type', default='CE', type=str, choices=['CE', 'BPR', 'BCE', 'CUSTOM'])

    return parser

