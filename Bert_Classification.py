##############################################################
#
# Bert_Classification.py
# This file contains the code for fine-tuning BERT using a
# simple classification head.
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


class Bert_Classification_Model(nn.Module):
    """ A Model for bert fine tuning """

    def __init__(self):
        super(Bert_Classification_Model, self).__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        # self.bert_drop=nn.Dropout(0.2)
        # self.fc=nn.Linear(768,256)
        # self.out=nn.Linear(256,10)
        self.out = nn.Linear(768, 10)
        # self.relu=nn.ReLU()

    def forward(self, ids, mask, token_type_ids):
        """ Define how to perfom each call

        Parameters
        __________
        ids: array
            -
        mask: array
            - 
        token_type_ids: array
            -

        Returns
        _______
            - 
        """

        _, pooled_out = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids)
        # rh=self.bert_drop(pooled_out)
        # rh=self.fc(rh)
        # rh=self.relu(rh)
        return self.out(pooled_out)
