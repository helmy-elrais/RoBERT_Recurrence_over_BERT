##############################################################
#
# BERT_Hierarchical.py
# This file contains the code to fine-tune BERT by computing
# segment tensors as a pooled result from all the segments
# obtained after tokenization.
#
##############################################################
import pandas as pd
import numpy as np
import time
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler

import transformers
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
# get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup


class BERT_Hierarchical_Model(nn.Module):

    def __init__(self, pooling_method="mean"):
        super(BERT_Hierarchical_Model, self).__init__()

        self.pooling_method = pooling_method

        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.out = nn.Linear(768, 10)

    def forward(self, ids, mask, token_type_ids, lengt):

        _, pooled_out = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids)

        chunks_emb = pooled_out.split_with_sizes(lengt)

        if self.pooling_method == "mean":
            emb_pool = torch.stack([torch.mean(x, 0) for x in chunks_emb])
        elif self.pooling_method == "max":
            emb_pool = torch.stack([torch.max(x, 0)[0] for x in chunks_emb])

        return self.out(emb_pool)
