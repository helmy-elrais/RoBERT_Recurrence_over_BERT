##############################################################
#
# RoBERT.py
# This file contains the implementation of the RoBERT model
# An LSTM is applied to a segmented document. The resulting
# embedding is used for document-level classification
#
##############################################################
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
# get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time


class RoBERT_Model(nn.Module):
    """ Make an LSTM model over a fine tuned bert model.

    Parameters
    __________
    bertFineTuned: BertModel
        A bert fine tuned instance

    """

    def __init__(self, bertFineTuned):
        super(RoBERT_Model, self).__init__()
        self.bertFineTuned = bertFineTuned
        self.lstm = nn.LSTM(768, 100, num_layers=1, bidirectional=False)
        self.out = nn.Linear(100, 10)

    def forward(self, ids, mask, token_type_ids, lengt):
        """ Define how to performed each call

        Parameters
        __________
        ids: array
            -
        mask: array
            - 
        token_type_ids: array
            -
        lengt: int
            -

        Returns:
        _______
        -
        """
        _, pooled_out = self.bertFineTuned(
            ids, attention_mask=mask, token_type_ids=token_type_ids)
        chunks_emb = pooled_out.split_with_sizes(lengt)

        seq_lengths = torch.LongTensor([x for x in map(len, chunks_emb)])

        batch_emb_pad = nn.utils.rnn.pad_sequence(
            chunks_emb, padding_value=-91, batch_first=True)
        batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        lstm_input = nn.utils.rnn.pack_padded_sequence(
            batch_emb, seq_lengths.cpu().numpy(), batch_first=False, enforce_sorted=False)

        packed_output, (h_t, h_c) = self.lstm(lstm_input, )  # (h_t, h_c))
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, padding_value=-91)

        h_t = h_t.view(-1, 100)

        return self.out(h_t)
