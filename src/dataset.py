"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import sys 
#For loading large text corpora
maxInt = sys.maxsize
maxInt = int(maxInt/10)
decrement = True
while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.dict_index = {}
        for index, word in enumerate(self.dict):
            self.dict_index[word] = index
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        texts, labels = [], []
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
        document_encode = [
            [self.dict_index.get(word,-1) for word in word_tokenize(text=sentences)] for sentences
            in sent_tokenize(text=text)]
        
        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1
        return document_encode


    def __getitem__(self, index):
        label = self.labels[index]
        text_1 = self.texts[index][0]
        text_2 = self.texts[index][1]
        return text_1.astype(np.int64), text_2.astype(np.int64) ,label

if __name__ == '__main__':
    test = MyDataset(data_path="../data/test_pair.csv", dict_path="../data/glove.6B.50d.txt")
    print(test.__getitem__(index=1)[0].shape)
    print(test.__getitem__(index=1)[1].shape)
    
