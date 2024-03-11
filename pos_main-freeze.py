
from pathlib import Path
import re
import sys
from transformers import AutoTokenizer, BertForTokenClassification, AdamW
from torch.utils.data import DataLoader
import torch
import numpy as np
import gc
from sklearn.metrics import classification_report, f1_score
import conllu
import pickle
import time
import torch.nn as nn
from torch.optim import Adam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
    
def Read_Words_and_Labels(path):
    tree=readconll(path)
    texts=[]
    tags=[]
    i=0
    for sentence in tree:
        if(len(sentence)<16):
            texts.append([])
            tags.append([])
            for token in sentence:
                tags[i].append(token['upos'])
                texts[i].append(str(token))
            i+=1
    return texts, tags


def MakeLabel2id_dict(dataset):
    vocab={}
    i=0
    for tags in dataset:
        for tag in tags:
            if tag not in vocab and tag!='_':
                vocab.update({tag : i})
                i+=1
    print(vocab)
    return vocab, {y:x for x,y in vocab.items()}


def encode_tags(texts, tags, encodings, tokenizer):
    encoded_labels = []
    for i, wordlist in enumerate(texts):
        doc_enc_labels = np.ones(len(encodings['input_ids'][1]),dtype=int) * -100
        k=0
        for u, word in enumerate(wordlist):
            length=len(tokenizer(word, add_special_tokens=False)['input_ids'])
            if not tags[i][u] == '_':
                try:
                    doc_enc_labels[k+length] = label2id_dict[tags[i][u]]
                except IndexError:
                    pass
            k+=length
        encoded_labels.append(doc_enc_labels)
    return encoded_labels


def readconll(path):
    with open(path) as conll_data:
        trees = conllu.parse(conll_data.read())
        return trees


    
    
class CusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def PrepareForSeqeval(LabelBatch, PredictionBatch):
    LabelRetList=[]
    PredRetList=[]
    for i, Labels in enumerate(zip(LabelBatch.tolist(), PredictionBatch.tolist())):
        PredRetList.append([])
        LabelRetList.append([])
        for label in zip(Labels[0], Labels[1]):
            if label[0] != -100:
                LabelRetList[i].append(id2label_dict[label[0]])
                PredRetList[i].append(id2label_dict[label[1]])
    return LabelRetList, PredRetList


def AddLabelsTogether(RetList, Labels):
    for label_list in Labels:
        RetList.append(label_list[0])
    return RetList


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


def RUN(path, modelname):
    start_time = time.perf_counter()
    train_texts, train_tags = Read_Words_and_Labels("DATA/POS_dataset/" + path[0:2]+"/" + path + "-ud-train.conllu")
    eval_texts, eval_tags = Read_Words_and_Labels("DATA/POS_dataset/" + path[0:2]+"/" + path+"-ud-test.conllu")
    
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    
    train_encodings = tokenizer(train_texts, is_split_into_words=True,  padding=True, truncation=True, max_length=50)
    eval_encodings = tokenizer(eval_texts, is_split_into_words=True, padding=True, truncation=True, max_length=50)
    
    global label2id_dict, id2label_dict
    label2id_dict, id2label_dict = MakeLabel2id_dict(train_tags+eval_tags)

    model = BertForTokenClassification.from_pretrained(modelname, num_labels=len(label2id_dict), return_dict=True)
    classifier = SimpleClassifier(output_dim =len(label2id_dict))
    model.classifier = classifier
    
    
    train_labels = encode_tags(train_texts, train_tags, train_encodings, tokenizer)
    eval_labels = encode_tags(eval_texts, eval_tags, eval_encodings, tokenizer)
    train_dataset = CusDataset(train_encodings, train_labels)
    val_dataset = CusDataset(eval_encodings, eval_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    for param in model.bert.parameters():
        param.requires_grad = False

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader=DataLoader(val_dataset, batch_size=16, shuffle=False)

    optim = Adam(model.parameters(), lr=1e-3)
    best_acc = 0
    for epoch in range(10):

        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].type(torch.int64).to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

            
        Accs={}
        model.eval()
        total=0
        correct=0
        total_true=[]
        total_pred=[]
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].type(torch.int64).to(device)

            outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs[0]
            
            predicted=torch.argmax(outputs.logits, dim = 2)
            
            correct += predicted.eq(labels).sum().item()
            total += (~labels.eq(-100)).sum().item()
            y_true, y_pred=PrepareForSeqeval(labels, predicted)
            total_true=AddLabelsTogether(total_true, y_true)
            total_pred=AddLabelsTogether(total_pred, y_pred)
            
        Total_accs = classification_report(total_true, total_pred, output_dict=True)
        Accuracy = correct / total    

        if best_acc<Accuracy:
            best_acc = Accuracy
            best_total_accs = Total_accs
            best_model = model
        gc.collect()
        torch.cuda.empty_cache()
        print(str(epoch) + ". Epoch")
    os.makedirs('POS_result', exist_ok=True)
    f=open('POS_result/'+path + '_' + modelname.replace("/", "_") + "_prec_and_rec_freeze.pkl", "wb")
    pickle.dump(best_total_accs, f)
    f.close()
    torch.save(best_model, 'POS_result/'+path + '_' + modelname.replace("/", "_") + "freeze.pth")
    end_time = time.perf_counter()
    run_time = end_time - start_time
    modelname = 'Multilingual' if modelname == "bert-base-multilingual-cased" else 'Monolingual'
    best_acc = best_acc*100
    return str('%.3f' % best_acc), path[:2], modelname, run_time
    
if __name__ == "__main__":
    conf_languages={'de_gsd' : 'bert-base-german-cased',
                    'en_gum' : 'bert-base-cased',
                    'fr_gsd' : 'Geotrend/bert-base-fr-cased',
                    'hi_hdtb': 'Geotrend/bert-base-hi-cased',
                    'ja_gsd' : 'cl-tohoku/bert-base-japanese-v2',
                    'ko_gsd' : 'kykim/bert-kor-base',
                    'tr_boun': 'dbmdz/bert-base-turkish-cased',
                    'zh_gsd' : 'bert-base-chinese',
                    'hu_szeged' : 'SZTAKI-HLT/hubert-base-cc'}
    default="bert-base-multilingual-cased"
    Results = []
    for key in conf_languages:
        Results.append(RUN(key, conf_languages[key]))
        Results.append(RUN(key, default))
    df = pd.DataFrame(Results, columns =['Accuracy', 'Language', 'Model', 'Time'], dtype = float)
    
    
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df, kind="bar",
        x="Language", y="Accuracy", hue="Model",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Accuracy (%)")
    plt.savefig('Pos_Accuracy_freeze.png')
    
    g = sns.catplot(
        data=df, kind="bar",
        x="Language", y="Time", hue="Model",
        ci="sd", palette="Dark2", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Time (s)")
    plt.savefig('Pos_Time_freeze.png')

