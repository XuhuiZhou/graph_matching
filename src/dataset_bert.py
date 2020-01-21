"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
from torch.utils.data.dataset import Dataset
import torch
import csv
import numpy as np
import sys
import os
from transformers import *
#For loading large text corpora
os.environ["CUDA_VISIBLE_DEVICES"]="0"
class MyDataset(Dataset):
    def __init__(self, data_path, model, tokenizer):
        super(MyDataset, self).__init__()
        texts, labels = [], []
        self.model = model
        self.tokenizer = tokenizer
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text_1 = ""
                text_2 = ""
                line[1] = line[1].replace('\001', '')
                line[2] = line[2].replace('\001', '')
                for tx in line[1].split():
                    text_1 += tx.lower()
                    text_1 += " "
                for tx in line[2].split():
                    text_2 += tx.lower()
                    text_2 += " "
                label = int(line[0])
                text_1 = self.process(text_1)
                text_2 = self.process(text_2)
                texts.append((text_1, text_2))
                labels.append(label)
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def process(self, text):
        input_ids = self.tokenizer.encode(text, add_special_tokens=True, max_length=500)
        cls_ = self.bert_represent(input_ids)
        return cls_

    def bert_represent(self, input_ids):
        # Tokenized input
        # text = "[CLS] I got restricted because Tom reported my reply [SEP]"
        # token_type_ids = torch.tensor(token_type_ids).unsqueeze(0)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to('cuda')
        with torch.no_grad():
            outputs = self.model(input_ids)
        last_hidden_states = outputs[0][0][0]  # The last hidden-state is the first element of the output tuple
        return last_hidden_states.cpu()

    def __getitem__(self, index):
        label = self.labels[index]
        text_1 = self.texts[index][0]
        text_2 = self.texts[index][1]
        #cls_ = self.process(text_1, text_2)
        cls1 = text_1.numpy()
        cls2 = text_2.numpy()
        return cls1, cls2, label

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to('cuda')
    bert_model.eval()
    test = MyDataset(data_path="../data/test_pair.csv", model= bert_model, tokenizer= tokenizer)
    print(test.__getitem__(index=1)[0].shape)
    print(test.__getitem__(index=1)[1].shape)