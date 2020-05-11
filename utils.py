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


def loss_fun(outputs, targets):
    loss= nn.CrossEntropyLoss()
    return loss(outputs, targets)
    #return nn.BCEWithLogitsLoss()(outputs, targets)   


def train_loop_fun(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    t0 = time.time()
    losses=[]
    for batch_idx, data in enumerate(data_loader):
        #for datapoint in data:
        for sentence_tokens in data:
            for chunk_tokens in sentence_tokens:
            
    #             model.half()

                ids=chunk_tokens["ids"]
                mask=chunk_tokens["mask"]
                token_type_ids = chunk_tokens["token_type_ids"]
                targets = chunk_tokens["targets"]
                
                ids=torch.unsqueeze(ids,dim=0)
                mask=torch.unsqueeze(mask, dim=0)
                token_type_ids=torch.unsqueeze(token_type_ids, dim=0)
                targets=torch.unsqueeze(targets, dim=0)
                
    #         for i in range(len(data["targets"])):
    #             ids=torch.cat(data["ids"])
    #             mask=torch.cat(data["mask"])
    #             token_type_ids = torch.cat(data["token_type_ids"])
    #             targets = torch.cat(data["targets"])
                #print(targets)

    #             ids=data["ids"][i]
    #             mask=data["mask"][i]
    #             token_type_ids = data["token_type_ids"][i]
    #             targets = data["targets"][i]

                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.long)

                optimizer.zero_grad()
                outputs=model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                loss=loss_fun(outputs, targets)
                loss.backward()
                model.float()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                losses.append(loss.item())
        if batch_idx % 10 == 0:
            print(f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            t0 = time.time()
    return losses


def eval_loop_fun(data_loader, model, device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    losses=[]
    for batch_idx, data in enumerate(data_loader):
        
#         for i in range(len(data["targets"])):
        
        for sentence_tokens in data:
            for chunk_tokens in sentence_tokens:
                ids=chunk_tokens["ids"]
                mask=chunk_tokens["mask"]
                token_type_ids = chunk_tokens["token_type_ids"]
                targets = chunk_tokens["targets"]

                ids=torch.unsqueeze(ids,dim=0)
                mask=torch.unsqueeze(mask, dim=0)
                token_type_ids=torch.unsqueeze(token_type_ids, dim=0)
                targets=torch.unsqueeze(targets, dim=0)
                
    #             ids=data["ids"][i]
    #             mask=data["mask"][i]
    #             token_type_ids = data["token_type_ids"][i]
    #             targets = data["targets"][i]
                #print(targets)

                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.long)
                with torch.no_grad():
                    outputs=model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                    loss=loss_fun(outputs, targets)
                    losses.append(loss.item())

                fin_targets.append(targets.cpu().detach().numpy())
                fin_outputs.append(torch.softmax(outputs, dim=1).cpu().detach().numpy())
    return np.vstack(fin_outputs), np.vstack(fin_targets), losses



def train_loop_fun1(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    t0 = time.time()
    losses=[]
    for batch_idx, batch in enumerate(data_loader):
        
#         model.half()
#         ids_batch=[data["ids"] for data in batch]
#         mask_batch=[data["mask"] for data in batch]
#         token_type_ids_batch = [data["token_type_ids"] for data in batch]
#         targets_batch = [data["targets"] for data in batch]
#         lengt_batch=[data['len'] for data in batch]
        ids=[data["ids"] for data in batch]
        mask=[data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch]
        lengt=[data['len'] for data in batch]
        
        ids=torch.cat(ids)
        mask=torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.cat(targets)
        lengt=torch.cat(lengt)

#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]
            
        
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs=model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss=loss_fun(outputs, targets)
        loss.backward()
        model.float()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        if batch_idx % 10 == 0:
            print(f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            t0 = time.time()
    return losses




def eval_loop_fun1(data_loader, model, device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    losses=[]
    for batch_idx, batch in enumerate(data_loader):
        
#         model.half()
        ids=[data["ids"] for data in batch]
        mask=[data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch]
        lengt=[data['len'] for data in batch]
#         ids_batch=[data["ids"] for data in batch]
#         mask_batch=[data["mask"] for data in batch]
#         token_type_ids_batch = [data["token_type_ids"] for data in batch]
#         targets_batch = [data["targets"] for data in batch]
#         lengt_batch=[data['len'] for data in batch]
        
#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]
            
        ids=torch.cat(ids)
        mask=torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.cat(targets)
        lengt=torch.cat(lengt)

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        with torch.no_grad():
            outputs=model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss=loss_fun(outputs, targets)
            losses.append(loss.item())

        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(outputs, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses        
#     return np.vstack(fin_outputs), np.vstack(fin_targets), losses






def my_collate(batch):
    #print("$$$$$$$$$$$$$$$",batch)
#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
    #target = torch.LongTensor(target)
    return batch


def my_collate1(batches):
    #return batches
    return [{key:torch.stack(value) for key, value in batch.items()} for batch in batches]



def evaluate(target, predicted):
    true_label_mask=[1 if (np.argmax(x)-target[i])==0 else 0 for i,x in enumerate(predicted)]
    nb_prediction=len(true_label_mask)
    true_prediction=sum(true_label_mask)
    false_prediction=nb_prediction-true_prediction
    accuracy= true_prediction/nb_prediction
    return{
        "accuracy":accuracy,
        "nb exemple":len(target),
        "true_prediction":true_prediction,
        "false_prediction":false_prediction,
    }


def rnn_train_loop_fun1(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    t0 = time.time()
    losses=[]
    for batch_idx, batch in enumerate(data_loader):
#         model.half()
#         ids_batch=[data["ids"] for data in batch]
#         mask_batch=[data["mask"] for data in batch]
#         token_type_ids_batch = [data["token_type_ids"] for data in batch]
#         targets_batch = [data["targets"] for data in batch]
#         lengt_batch=[data['len'] for data in batch]
        ids=[data["ids"] for data in batch]
        mask=[data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"][0] for data in batch]
        lengt=[data['len'] for data in batch]
        
        ids=torch.cat(ids)
        mask=torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.stack(targets)
        lengt=torch.cat(lengt)
        lengt=[x.item() for x in lengt]        

#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]
            
        
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs=model(ids=ids, mask=mask, token_type_ids=token_type_ids, lengt=lengt)
        loss=loss_fun(outputs, targets)
        loss.backward()
        model.float()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        if batch_idx % 10 == 0:
            print(f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            t0 = time.time()
    return losses



def rnn_eval_loop_fun1(data_loader, model, device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    losses=[]
    for batch_idx, batch in enumerate(data_loader):
        
#         model.half()
        ids=[data["ids"] for data in batch]
        mask=[data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"][0] for data in batch]
        lengt=[data['len'] for data in batch]
#         ids_batch=[data["ids"] for data in batch]
#         mask_batch=[data["mask"] for data in batch]
#         token_type_ids_batch = [data["token_type_ids"] for data in batch]
#         targets_batch = [data["targets"] for data in batch]
#         lengt_batch=[data['len'] for data in batch]
        
#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]
            
        ids=torch.cat(ids)
        mask=torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.stack(targets)
        lengt=torch.cat(lengt)
        lengt=[x.item() for x in lengt] 

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        with torch.no_grad():
            outputs=model(ids=ids, mask=mask, token_type_ids=token_type_ids, lengt=lengt)
            loss=loss_fun(outputs, targets)
            losses.append(loss.item())

        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(outputs, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses        
#     return np.vstack(fin_outputs), np.vstack(fin_targets), losses