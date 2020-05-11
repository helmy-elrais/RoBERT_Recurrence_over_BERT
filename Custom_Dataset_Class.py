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
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW# get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
import time

class ConsumerComplaintsDataset(Dataset):
    
    
    def __init__(self, tokenizer, max_len,chunk_len=200, overlap_len=50, approach="long_terms", extremum=100, max_size_dataset=None,file_location="./us-consumer-finance-complaints/consumer_complaints.csv",min_len=249):
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.overlap_len=overlap_len
        self.chunk_len=chunk_len
        self.approach=approach
        self.extremum=extremum
        self.min_len=min_len
        self.max_size_dataset=max_size_dataset
        self.data, self.label = self.process_data(file_location,)
        
    
    def process_data(self,file_location):
        # Load the dataset into a pandas dataframe.
        print("Nettoyage des données")
        df=pd.read_csv(file_location,dtype="unicode")
        train_raw = df[df.consumer_complaint_narrative.notnull()]
        train_raw = train_raw.assign(len_txt=train_raw.consumer_complaint_narrative.apply(lambda x: len(x.split())))
        train_raw = train_raw[train_raw.len_txt >self.min_len]
        train_raw = train_raw[['consumer_complaint_narrative', 'product']]
        train_raw.reset_index(inplace=True, drop=True)
        train_raw.at[train_raw['product'] == 'Credit reporting', 'product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
        train_raw.at[train_raw['product'] == 'Credit card', 'product'] = 'Credit card or prepaid card'
        train_raw.at[train_raw['product'] == 'Prepaid card', 'product'] = 'Credit card or prepaid card'
        train_raw.at[train_raw['product'] == 'Payday loan', 'product'] = 'Payday loan, title loan, or personal loan'
        train_raw.at[train_raw['product'] == 'Virtual currency', 'product'] = 'Money transfer, virtual currency, or money service'
        train_raw=train_raw.rename(columns = {'consumer_complaint_narrative':'text', 'product':'label'})
        LE = LabelEncoder()
        train_raw['label'] = LE.fit_transform(train_raw['label'])
        train = train_raw.copy()
        if(self.max_size_dataset) : train=train.loc[0:self.max_size_dataset,:]
        train = train.reindex(np.random.permutation(train.index))
        train['text']  = train.text.apply(self.clean_txt)
        return train['text'].values, train['label'].values
    
    def clean_txt(self,text):
        text = re.sub("'", "",text)
        text=re.sub("(\\W)+"," ",text)    
        return text
    
    def technique(self, text):
        if self.approach=="rnn":
            return self.long_terms_tokenizer
        elif self.approach=="first":
            return text[:self.extremum]
        elif self.approach=="last":
            return text[-self.extremum:]
        elif self.approach=="mixt":
            return text[:self.extremum//2] + text[-self.extremum//2:]
    
    
    def long_terms_tokenizer(self, data_tokenize, targets):

        long_terms_token=[]

        previous_input_ids=data_tokenize["input_ids"].reshape(-1)
        previous_attention_mask=data_tokenize["attention_mask"].reshape(-1)
        previous_token_type_ids=data_tokenize["token_type_ids"].reshape(-1)
        remain= data_tokenize.get("overflowing_tokens")
        targets=torch.tensor(targets, dtype=torch.int)

        long_terms_token.append({
                'ids': previous_input_ids,
                'mask': previous_attention_mask,
                'token_type_ids': previous_token_type_ids,
                'targets':targets})

        if remain:
            remain=torch.tensor(remain, dtype=torch.long)
            idxs=range(len(remain)+self.chunk_len)
            idxs=idxs[(self.chunk_len-self.overlap_len-2)::(self.chunk_len-self.overlap_len-2)]
            input_ids_first_overlap=previous_input_ids[-(self.overlap_len+1):-1]
            start_token=torch.tensor([101], dtype=torch.long)
            end_token=torch.tensor([102], dtype=torch.long)

            for i,idx in enumerate(idxs):
                if i==0:
                    input_ids=torch.cat((input_ids_first_overlap,remain[:idx]))
                elif i==len(idxs):
                    input_ids=remain[idx:]
                elif previous_idx>=len(remain):
                    break
                else:
                    input_ids=remain[(previous_idx-self.overlap_len):idx]

                previous_idx=idx

                nb_token=len(input_ids)+2
                attention_mask=torch.ones(self.chunk_len)        
                attention_mask[nb_token:self.chunk_len]=0
                token_type_ids=torch.zeros(self.chunk_len)
                input_ids=torch.cat((start_token,input_ids,end_token))
                if self.chunk_len-nb_token>0:
                    padding=torch.zeros(self.chunk_len-nb_token, dtype=torch.long)
                    input_ids=torch.cat((input_ids,padding))

                long_terms_token.append({
                    'ids': input_ids,#torch.tensor(ids, dtype=torch.long),
                    'mask': attention_mask,#torch.tensor(mask, dtype=torch.long),
                    'token_type_ids': token_type_ids,#torch.tensor(token_type_ids, dtype=torch.long),
                    'targets':targets
                })

        return long_terms_token
    
    def __getitem__(self, idx):
        consumer_complaint=str(self.data[idx])
        targets=int(self.label[idx])
        data=self.tokenizer.encode_plus(
            consumer_complaint,
            max_length=self.chunk_len,
            pad_to_max_length = True,
            add_special_tokens=True,
            return_attention_mask = True,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_tensors = 'pt')
        
        #ids=data["input_ids"]
        #token_type_ids=data["token_type_ids"]
        #mask=data["attention_mask"]
        
        #padding_len=self.max_len - len(ids)
        #ids=ids+([0]*padding_len)
        #token_type_ids=token_type_ids+([0]*padding_len)
        #mask=mask+([0]*padding_len)
        
        long_token=self.long_terms_tokenizer(data, targets)
        return long_token
#         return {
#             'ids': data["input_ids"].reshape(-1),#torch.tensor(ids, dtype=torch.long),
#             'mask': data["attention_mask"].reshape(-1),#torch.tensor(mask, dtype=torch.long),
#             'token_type_ids': data["token_type_ids"].reshape(-1),#torch.tensor(token_type_ids, dtype=torch.long),
#             'targets':torch.tensor(targets, dtype=torch.int)
#         }
        
    def __len__(self):
        return self.label.shape[0]


class ConsumerComplaintsDataset1(Dataset):
    
    
    def __init__(self, tokenizer, max_len,chunk_len=200, overlap_len=50, approach="long_terms", extremum=100, max_size_dataset=None,file_location="./us-consumer-finance-complaints/consumer_complaints.csv",min_len=249):
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.overlap_len=overlap_len
        self.chunk_len=chunk_len
        self.approach=approach
        self.extremum=extremum
        self.min_len=min_len
        self.max_size_dataset=max_size_dataset
        self.data, self.label = self.process_data(file_location,)
        
    
    def process_data(self,file_location):
        # Load the dataset into a pandas dataframe.
        print("Nettoyage des données")
        df=pd.read_csv(file_location,dtype="unicode")
        train_raw = df[df.consumer_complaint_narrative.notnull()]
        train_raw = train_raw.assign(len_txt=train_raw.consumer_complaint_narrative.apply(lambda x: len(x.split())))
        train_raw = train_raw[train_raw.len_txt >self.min_len]
        train_raw = train_raw[['consumer_complaint_narrative', 'product']]
        train_raw.reset_index(inplace=True, drop=True)
        train_raw.at[train_raw['product'] == 'Credit reporting', 'product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
        train_raw.at[train_raw['product'] == 'Credit card', 'product'] = 'Credit card or prepaid card'
        train_raw.at[train_raw['product'] == 'Prepaid card', 'product'] = 'Credit card or prepaid card'
        train_raw.at[train_raw['product'] == 'Payday loan', 'product'] = 'Payday loan, title loan, or personal loan'
        train_raw.at[train_raw['product'] == 'Virtual currency', 'product'] = 'Money transfer, virtual currency, or money service'
        train_raw=train_raw.rename(columns = {'consumer_complaint_narrative':'text', 'product':'label'})
        LE = LabelEncoder()
        train_raw['label'] = LE.fit_transform(train_raw['label'])
        train = train_raw.copy()
        if(self.max_size_dataset) : train=train.loc[0:self.max_size_dataset,:]
        train = train.reindex(np.random.permutation(train.index))
        train['text']  = train.text.apply(self.clean_txt)
        return train['text'].values, train['label'].values
    
    def clean_txt(self,text):
        text = re.sub("'", "",text)
        text=re.sub("(\\W)+"," ",text)    
        return text
    
    def technique(self, text):
        if self.approach=="rnn":
            return self.long_terms_tokenizer
        elif self.approach=="first":
            return text[:self.extremum]
        elif self.approach=="last":
            return text[-self.extremum:]
        elif self.approach=="mixt":
            return text[:self.extremum//2] + text[-self.extremum//2:]
    
    
    def long_terms_tokenizer(self, data_tokenize, targets):

        long_terms_token=[]
        input_ids_list=[]
        attention_mask_list=[]
        token_type_ids_list=[]
        targets_list=[]
        
        previous_input_ids=data_tokenize["input_ids"].reshape(-1)
        previous_attention_mask=data_tokenize["attention_mask"].reshape(-1)
        previous_token_type_ids=data_tokenize["token_type_ids"].reshape(-1)
        remain= data_tokenize.get("overflowing_tokens")
        targets=torch.tensor(targets, dtype=torch.int)

#         long_terms_token.append({
#                 'ids': previous_input_ids,for key, value in d.items()
#                 'mask': previous_attention_mask,
#                 'token_type_ids': previous_token_type_ids,
#                 'targets':targets})
        input_ids_list.append(previous_input_ids)
        attention_mask_list.append(previous_attention_mask)
        token_type_ids_list.append(previous_token_type_ids)
        targets_list.append(targets)

        if remain:
            remain=torch.tensor(remain, dtype=torch.long)
            idxs=range(len(remain)+self.chunk_len)
            idxs=idxs[(self.chunk_len-self.overlap_len-2)::(self.chunk_len-self.overlap_len-2)]
            input_ids_first_overlap=previous_input_ids[-(self.overlap_len+1):-1]
            start_token=torch.tensor([101], dtype=torch.long)
            end_token=torch.tensor([102], dtype=torch.long)

            for i,idx in enumerate(idxs):
                if i==0:
                    input_ids=torch.cat((input_ids_first_overlap,remain[:idx]))
                elif i==len(idxs):
                    input_ids=remain[idx:]
                elif previous_idx>=len(remain):
                    break
                else:
                    input_ids=remain[(previous_idx-self.overlap_len):idx]

                previous_idx=idx

                nb_token=len(input_ids)+2
                attention_mask=torch.ones(self.chunk_len, dtype=torch.long)        
                attention_mask[nb_token:self.chunk_len]=0
                token_type_ids=torch.zeros(self.chunk_len, dtype=torch.long)
                input_ids=torch.cat((start_token,input_ids,end_token))
                if self.chunk_len-nb_token>0:
                    padding=torch.zeros(self.chunk_len-nb_token, dtype=torch.long)
                    input_ids=torch.cat((input_ids,padding))

#                 long_terms_token.append({
#                     'ids': input_ids,#torch.tensor(ids, dtype=torch.long),
#                     'mask': attention_mask,#torch.tensor(mask, dtype=torch.long),
#                     'token_type_ids': token_type_ids,#torch.tensor(token_type_ids, dtype=torch.long),
#                     'targets':targets
#                 })
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                token_type_ids_list.append(token_type_ids)
                targets_list.append(targets)
                
        return({
                    'ids': input_ids_list,#torch.tensor(ids, dtype=torch.long),
                    'mask': attention_mask_list,#torch.tensor(mask, dtype=torch.long),
                    'token_type_ids': token_type_ids_list,#torch.tensor(token_type_ids, dtype=torch.long),
                    'targets':targets_list,
                    'len':[torch.tensor(len(targets_list), dtype=torch.long)]
                })

#         return long_terms_token
    
    def __getitem__(self, idx):
        consumer_complaint=str(self.data[idx])
        targets=int(self.label[idx])
        data=self.tokenizer.encode_plus(
            consumer_complaint,
            max_length=self.chunk_len,
            pad_to_max_length = True,
            add_special_tokens=True,
            return_attention_mask = True,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_tensors = 'pt')
        
        #ids=data["input_ids"]
        #token_type_ids=data["token_type_ids"]
        #mask=data["attention_mask"]
        
        #padding_len=self.max_len - len(ids)
        #ids=ids+([0]*padding_len)
        #token_type_ids=token_type_ids+([0]*padding_len)
        #mask=mask+([0]*padding_len)
        
        long_token=self.long_terms_tokenizer(data, targets)
        return long_token
#         return {
#             'ids': data["input_ids"].reshape(-1),#torch.tensor(ids, dtype=torch.long),
#             'mask': data["attention_mask"].reshape(-1),#torch.tensor(mask, dtype=torch.long),
#             'token_type_ids': data["token_type_ids"].reshape(-1),#torch.tensor(token_type_ids, dtype=torch.long),
#             'targets':torch.tensor(targets, dtype=torch.int)
#         }
        
    def __len__(self):
        return self.label.shape[0]
