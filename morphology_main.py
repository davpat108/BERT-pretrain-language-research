from transformers import AdamW
from transformers import AutoTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
import torch
import pickle
import gc
from sklearn.metrics import classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

def read_tsv_with_pandas(language):
    if language == 'Fr':
        train_tsv=pd.read_csv('Morph_Data/gender/'+language+'/train.tsv',na_filter=None, quoting=3, sep="\t")
        dev_tsv =pd.read_csv('Morph_Data/gender/'+language+'/dev.tsv',na_filter=None, quoting=3, sep="\t")
    else:
        train_tsv=pd.read_csv('Morph_Data/case_noun/'+language+'/train.tsv',na_filter=None, quoting=3, sep="\t")
        dev_tsv =pd.read_csv('Morph_Data/case_noun/'+language+'/dev.tsv',na_filter=None, quoting=3, sep="\t")
        
    return train_tsv, dev_tsv


def get_data(tsv):
    data = []
    for obj in tsv.values:
        data_piece = []
        data_piece.append(obj[0])
        data_piece.append(obj[2])
        data_piece.append(obj[3])
        data.append(data_piece)
    return data


def make_label_and_id_dicts(data):
    label2id = {}
    id2label = {}
    i=0
    for sentence, number, case in data:
        if case not in label2id.keys():
            label2id.update({case:i})
            id2label.update({i:case})
            i+=1
    return label2id, id2label


def divide_to_texts_and_tags(data):
    texts = []
    tags = []
    for text, index, tag in data:
        texts.append(text.split())
        tags.append([index, tag])
    return texts, tags


def encode_tags(texts, tags, encodings, tokenizer):
    encoded_labels = []
    for i, wordlist in enumerate(texts):
        doc_enc_labels = np.ones(len(encodings['input_ids'][1]),dtype=int) * -100
        extra_length = 0
        for u in range(tags[i][0]):
            length = len(tokenizer(wordlist[u], add_special_tokens=False)['input_ids'])
            extra_length += length
        doc_enc_labels[extra_length+1] = label2id[tags[i][1]]
        encoded_labels.append(doc_enc_labels)
    return encoded_labels



def PrepareForSeqeval(LabelBatch, PredictionBatch):
    LabelRetList=[]
    PredRetList=[]
    for i, Labels in enumerate(zip(LabelBatch.tolist(), PredictionBatch.tolist())):
        PredRetList.append([])
        LabelRetList.append([])
        for label in zip(Labels[0], Labels[1]):
            if label[0] != -100:
                LabelRetList[i].append(id2label[label[0]])
                PredRetList[i].append(id2label[label[1]])
    return LabelRetList, PredRetList


def AddLabelsTogether(RetList, Labels):
    for label_list in Labels:
        RetList.append(label_list[0])
    return RetList


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


def RUN(path, modelname):
    start_time = time.perf_counter()
    train_tsv, dev_tsv = read_tsv_with_pandas(path)
    train_data = get_data(train_tsv)
    dev_data = get_data(dev_tsv)
    
    global label2id, id2label
    label2id, id2label = make_label_and_id_dicts(train_data)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = BertForTokenClassification.from_pretrained(modelname, num_labels=len(label2id), return_dict=True)
    
    train_texts, train_tags = divide_to_texts_and_tags(train_data)
    dev_texts, dev_tags = divide_to_texts_and_tags(dev_data)
    train_encodings = tokenizer(train_texts, is_split_into_words=True, padding=True, truncation=True, max_length=100)
    dev_encodings = tokenizer(dev_texts, is_split_into_words=True, padding=True, truncation=True, max_length=100)

    train_labels = encode_tags(train_texts, train_tags, train_encodings, tokenizer)
    dev_labels = encode_tags(dev_texts, dev_tags, dev_encodings, tokenizer)
    
    train_dataset = CusDataset(train_encodings, train_labels)
    dev_dataset = CusDataset(dev_encodings, dev_labels)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model.to(device)
        
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader=DataLoader(dev_dataset, batch_size=16, shuffle=False)

    optim = AdamW(model.parameters(), lr=5e-5)
    

    total_time = 0
    best_f1 = 0
    for epoch in range(15):
        total_true=[]
        total_pred=[]
        Accs={}
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
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].type(torch.int64).to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            predicted=torch.argmax(outputs.logits, dim=2)
            y_true, y_pred=PrepareForSeqeval(labels, predicted)
            total_true=AddLabelsTogether(total_true, y_true)
            total_pred=AddLabelsTogether(total_pred, y_pred)
  
        Accs=classification_report(total_true, total_pred, output_dict=True)
        f1 = f1_score(total_true, total_pred, average='weighted')
        if best_f1<f1:
            best_f1 = f1
            best_accs = Accs
            best_model = model


    
    os.makedirs('Morph_results/first_sub', exist_ok=True)
    f=open('Morph_results/first_sub/'+path+'_'+modelname.replace("/", "_")+"morphology.pkl", "wb")
    pickle.dump(best_accs, f)
    f.close()
    torch.save(best_model, 'Morph_results/first_sub/'+path+'_'+modelname.replace("/", "_")+"morphology.pth")
    gc.collect()
    end_time = time.perf_counter()
    run_time =  end_time - start_time
    modelname = 'Multilingual' if modelname == "bert-base-multilingual-cased" else 'Monolingual'
    return best_f1*100, path, modelname, run_time

if __name__ == "__main__":
    print(torch.cuda.is_available())
    conf_languages={'De' : 'bert-base-german-cased',
                    'Fr' : 'Geotrend/bert-base-fr-cased',
                    'Hi' : 'Geotrend/bert-base-hi-cased',
                    'Tr' : 'dbmdz/bert-base-turkish-cased',
                    'Hu' : 'SZTAKI-HLT/hubert-base-cc'}
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
    plt.savefig('Morph_F1.png')
    
    g = sns.catplot(
        data=df, kind="bar",
        x="Language", y="Time", hue="Model",
        ci="sd", palette="Dark2", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Times (s)")
    plt.savefig('Morph_Time.png')









