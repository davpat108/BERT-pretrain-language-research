
from pathlib import Path
import re
import sys
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import torch
import numpy as np
import gc
from seqeval.metrics import classification_report
import conllu
import pickle
from datasets import load_dataset
import time
from torch.optim import Adam
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SimpleClassifier(nn.Module):
    def __init__(self, output_dim, input_dim=768, hidden_dim=50):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, X):
        h = self.input_layer(X)
        h = self.relu(h)
        out = self.output_layer(h)
        return out

def RUN(lang, modelname):
    start_time = time.perf_counter()
    dataset = load_dataset("amazon_reviews_multi", lang)
    
    train_texts = dataset['train']['review_body'][::10]
    eval_texts =  dataset['validation']['review_body'][::10]
    
    train_labels = [label-1 for label in dataset['train']['stars'][::10]]
    eval_labels = [label-1 for label in dataset['validation']['stars'][::10]]
    
    
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    
    train_encodings = tokenizer(train_texts, is_split_into_words=False,  padding=True, truncation=True, max_length=50)
    eval_encodings = tokenizer(eval_texts, is_split_into_words=False, padding=True, truncation=True, max_length=50)
    

    model = BertForSequenceClassification.from_pretrained(modelname, num_labels=1, return_dict=True)

    train_dataset = SeqDataset(train_encodings, train_labels)
    val_dataset = SeqDataset(eval_encodings, eval_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    classifier = SimpleClassifier(output_dim = 1)
    model.classifier = classifier
    
    model.to(device)

    for param in model.bert.parameters():
        param.requires_grad = False
    

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader=DataLoader(val_dataset, batch_size=8, shuffle=False)

    optim = Adam(model.parameters(), lr=1e-3)
    i=0
    
    for epoch in range(4):
        model.train()
            
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].type(torch.FloatTensor)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            
        model.eval()
        total=0
        squared_sum=0
        MSE=0
        Best_MSE = 1000000
        Best_acc = 0
        loss=0
        good=0
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].type(torch.FloatTensor)
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask = attention_mask, labels=labels)
            loss += int(outputs.loss)
            
        for i, batch in enumerate(val_loader):
            if i<125 or i>500:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].type(torch.FloatTensor)
                labels = labels.to(device)
            
                outputs = model(input_ids, attention_mask = attention_mask)
                predicted = torch.round(outputs.logits).reshape(1,labels.size()[0])//4
                labels = labels//4
                total += labels.size()[0]
                good += predicted.eq(labels).sum().item()
                print(labels, good, predicted)
                
        MSE = loss/len(val_loader)
        Acc = good/total
        gc.collect()
        torch.cuda.empty_cache()
        if MSE < Best_MSE:
            Best_MSE = MSE*1000
            Best_acc = Acc * 100
            torch.save(model, 'Sentiment_Result/'+lang+"_" + modelname.replace("/", "_") + "regression_freeze.pth")
    end_time = time.perf_counter()
    run_time = end_time - start_time
    modelname = 'Multilingual' if modelname == "bert-base-multilingual-cased" else 'Monolingual'
    return str('%.3f' % Best_MSE), str('%.3f' % Best_acc), lang, modelname, abs(run_time)

    
    
if __name__ == "__main__":
    conf_languages={'de' : 'bert-base-german-cased',
                    'en' : 'bert-base-cased',
                    'fr' : 'Geotrend/bert-base-fr-cased',
                    'ja' : 'cl-tohoku/bert-base-japanese-v2',
                    'zh' : 'bert-base-chinese'}
    default="bert-base-multilingual-cased"
    Results = []
    for key in conf_languages:
        Results.append(RUN(key, conf_languages[key]))
        Results.append(RUN(key, default))
    df = pd.DataFrame(Results, columns =['MSE', 'Accuracy',  'Language', 'Model', 'Time'], dtype = float)
    
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df, kind="bar",
        x="Language", y="MSE", hue="Model",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "MSE *1000")
    plt.savefig('Sent_Freeze_regression_MSE.png')
    
    g = sns.catplot(
        data=df, kind="bar",
        x="Language", y="Time", hue="Model",
        ci="sd", palette="Dark2", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Times (s)")
    plt.savefig('Sent__Freeze_regression_time.png')
    
