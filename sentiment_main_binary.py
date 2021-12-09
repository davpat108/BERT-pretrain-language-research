
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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    
    

def RUN(lang, modelname):
    start_time = time.perf_counter()
    dataset = load_dataset("amazon_reviews_multi", lang)
    
    train_texts = dataset['train']['review_body'][:5000:] + dataset['train']['review_body'][-5000::]
    eval_texts =  dataset['validation']['review_body'][:200:] + dataset['validation']['review_body'][-200::]
    
    train_labels = [label//3 for label in dataset['train']['stars'][:5000:] + dataset['train']['stars'][-5000::]]
    eval_labels = [label//3 for label in dataset['validation']['stars'][:200:] + dataset['validation']['stars'][-200::]]
    

    tokenizer = AutoTokenizer.from_pretrained(modelname)
    
    train_encodings = tokenizer(train_texts, is_split_into_words=False,  padding=True, truncation=True, max_length=50)
    eval_encodings = tokenizer(eval_texts, is_split_into_words=False, padding=True, truncation=True, max_length=50)
    

    model = BertForSequenceClassification.from_pretrained(modelname, num_labels=2, return_dict=True)

    train_dataset = SeqDataset(train_encodings, train_labels)
    val_dataset = SeqDataset(eval_encodings, eval_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader=DataLoader(val_dataset, batch_size=16, shuffle=False)

    optim = AdamW(model.parameters(), lr=5e-5)
    i=0
    total = 0
    good = 0
    for epoch in range(4):
        model.train()
            
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            predicted = torch.argmax(outputs.logits, dim = 1)
            total += labels.size()[0]
            good += predicted.eq(labels).sum().item()
            loss.backward()
            optim.step()
        Acc = good/total
        print("TRAIN ACC:", Acc)
        model.eval()
        total = 0
        good = 0
        Best_Acc = 0
        print("eval")
        loss=0
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask = attention_mask, labels=labels)
            predicted = torch.argmax(outputs.logits, dim = 1)
            #print(predicted, labels)
            total += labels.size()[0]
            good += predicted.eq(labels).sum().item()
        Acc = good/total
        print(Acc)
        gc.collect()
        torch.cuda.empty_cache()
        print(str(epoch) + ". Epoch")
        if Acc > Best_Acc:
            Best_Acc = Acc
            torch.save(model, 'Sentiment_Result/'+lang+"_" + modelname.replace("/", "_") + "binary.pth")
    end_time = time.perf_counter()
    run_time = end_time - start_time 
    modelname = 'Multilingual' if modelname == "bert-base-multilingual-cased" else 'Monolingual'
    Best_Acc = Best_Acc*100
    return str('%.3f' % Best_Acc), lang, modelname, abs(run_time)

    
    
if __name__ == "__main__":
    conf_languages={'de' : 'bert-base-german-cased',
                    'en' : 'bert-base-cased',
                    'fr' : 'Geotrend/bert-base-fr-cased',
                    'ja' : 'cl-tohoku/bert-base-japanese-v2',
                    'zh' : 'bert-base-chinese'}
    default="bert-base-multilingual-cased"
    Results = []
    for key in conf_languages:
        Results.append(RUN(key, default))
        Results.append(RUN(key, conf_languages[key]))
    df = pd.DataFrame(Results, columns =['Accuracy', 'Language', 'Model', 'Time'], dtype = float)
    
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df, kind="bar",
        x="Language", y="Accuracy", hue="Model",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Accuracy (%)")
    plt.savefig('Sent_binary_accuracy.png')
    g = sns.catplot(
        data=df, kind="bar",
        x="Language", y="Time", hue="Model",
        ci="sd", palette="Dark2", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Time (s)")
    plt.savefig('Sent_binary_time.png')

