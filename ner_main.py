from pathlib import Path
import re
from transformers import AdamW
from transformers import AutoTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
import gc
from seqeval.metrics import classification_report, f1_score
import time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

np.random.seed(122)
torch.manual_seed(122)


def Read_Words_and_Labels(file):
    file_path = Path(file)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token[3:])
            tags.append(tag)
        if(len(tags)<40):
            token_docs.append(tokens)
            tag_docs.append(tags)

    return token_docs, tag_docs


label2id_dict={     
    "B-PER" : 0,
    "I-PER" : 1,
    "O"     : 2,
    "B-ORG" : 3,
    "I-ORG" : 4,
    "B-LOC" : 5,
    "I-LOC" : 6}

id2label_dict={     
    0 : "B-PER",
    1 : "I-PER",
    2 : "O",
    3 : "B-ORG",
    4 : "I-ORG",
    5 : "B-LOC",
    6 : "I-LOC"}


def encode_tags(texts, tags, encodings, tokenizer):
    encoded_labels = []
    print("Train/Eval Encodins size with padding: " + str(len(encodings['input_ids'][1])))
    for i, wordlist in enumerate(texts):
        doc_enc_labels = np.ones(len(encodings['input_ids'][1]),dtype=int) * -100
        k=0
        for u, word in enumerate(wordlist):
            length=len(tokenizer(word, add_special_tokens=False)['input_ids'])
            try:
                doc_enc_labels[k+1]=label2id_dict[tags[i][u]]
            except IndexError:
                pass
            k+=length
        encoded_labels.append(doc_enc_labels)
    return encoded_labels




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
        RetList.append(label_list)
    return RetList

def RUN(path, modelname):
    start_time = time.perf_counter()
    train_texts, train_tags=Read_Words_and_Labels("DATA/NER/Wikiann/"+path+"/train")
    eval_texts, eval_tags=Read_Words_and_Labels("DATA/NER/Wikiann/"+path+"/dev")
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    train_encodings = tokenizer(train_texts, is_split_into_words=True,  padding=True, truncation=True, max_length=100)
    eval_encodings = tokenizer(eval_texts, is_split_into_words=True, padding=True, truncation=True, max_length=100)


    model = BertForTokenClassification.from_pretrained(modelname, num_labels=len(label2id_dict), return_dict=True)

    print(path, modelname)
    train_labels = encode_tags(train_texts, train_tags, train_encodings, tokenizer)
    eval_labels = encode_tags(eval_texts, eval_tags, eval_encodings, tokenizer)
    print(train_labels[0])


    train_dataset = CusDataset(train_encodings, train_labels)
    val_dataset = CusDataset(eval_encodings, eval_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    
        
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader=DataLoader(val_dataset, batch_size=16, shuffle=False)

    optim = AdamW(model.parameters(), lr=5e-5)

    total_time = 0
    best_F1 = 0
    for epoch in range(5):
        Accs={}
        total_true=[]
        total_pred=[]
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
        Accs={}
        model.eval()
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            predicted=torch.argmax(outputs.logits, dim=2)
            y_true, y_pred=PrepareForSeqeval(labels, predicted)
            total_true=AddLabelsTogether(total_true, y_true)
            total_pred=AddLabelsTogether(total_pred, y_pred)
        Accs=classification_report(total_true, total_pred, output_dict=True)
        f1 = f1_score(total_true, total_pred, average='weighted')
        if best_F1<f1:
            best_F1 = f1
            best_Accs = Accs
            best_model = model


    os.makedirs('NER_Result/NO_Freeze', exist_ok=True)
    f=open('NER_Result/NO_Freeze/'+path+'_'+modelname.replace("/", "_")+"_seqeval.pkl", "wb")
    pickle.dump(best_Accs, f)
    f.close()
    torch.save(best_model, 'NER_Result/NO_Freeze/'+path+'_'+modelname.replace("/", "_")+".pth")
    gc.collect()
    end_time = time.perf_counter()
    run_time =  end_time - start_time
    modelname = 'Multilingual' if modelname == "bert-base-multilingual-cased" else 'Monolingual'
    return best_F1*100, path, modelname, run_time
    
    
if __name__ == "__main__":
    print(torch.cuda.is_available())
    conf_languages={'de' : 'bert-base-german-cased',
                    'en' : 'bert-base-cased',
                    'fr' : 'Geotrend/bert-base-fr-cased',
                    'hi' : 'Geotrend/bert-base-hi-cased',
                    'ja' : 'cl-tohoku/bert-base-japanese-v2',
                    'ko' : 'kykim/bert-kor-base',
                    'tr' : 'dbmdz/bert-base-turkish-cased',
                    'zh' : 'bert-base-chinese'}
    default="bert-base-multilingual-cased"
    Results = []
    for key in conf_languages:
        Results.append(RUN(key, conf_languages[key]))
        Results.append(RUN(key, default))
    df = pd.DataFrame(Results, columns =['F1', 'Language', 'Model', 'Time'], dtype = float)
    
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df, kind="bar",
        x="Language", y="F1", hue="Model",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "F1-score (%)")
    plt.savefig('Ner_F1.png')
    
    g = sns.catplot(
        data=df, kind="bar",
        x="Language", y="Time", hue="Model",
        ci="sd", palette="Dark2", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Times (s)")

    plt.savefig('Ner_Time.png')
