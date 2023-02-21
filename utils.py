import logging
import torch
import random
from torch.utils.data import Dataset

class prepareDataset(Dataset):
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        input_ids, token_type_ids, attention_mask = self.prepare_inputs(self.cfg, self.dataset[item]["sentence"])
        label = self.prepare_labels(self.dataset[item]["label"])

        return input_ids, token_type_ids, attention_mask, label

    def prepare_inputs(self, CFG, text):

        tokens = CFG.tokenizer(text)

        if len(tokens["input_ids"]) > CFG.max_len:
            tokens["input_ids"] = tokens["input_ids"][0:CFG.max_len]
            tokens["token_type_ids"] = tokens["token_type_ids"][0:CFG.max_len]
            tokens["attention_mask"] = tokens["attention_mask"][0:CFG.max_len]
        else:
            while len(tokens["input_ids"]) < CFG.max_len:
                tokens["input_ids"].append(0)
                tokens["token_type_ids"].append(0)
                tokens["attention_mask"].append(0)

        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        token_type_ids = torch.tensor(tokens["token_type_ids"])

        return input_ids, token_type_ids, attention_mask

    def prepare_labels(self, label):
        return torch.tensor(label, dtype=torch.long)

def load_dataset(path):
    dataset = []

    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            if len(line) == 2:
                dataset.append({"sentence": line[0], "label": int(line[1])})
    return dataset


def cal(preds, labels, TP, FP, FN, ACC):
    for i in range(len(preds)):
        if preds[i]==labels[i]:
            ACC += 1
            TP[preds[i]] += 1
        else:
            FP[preds[i]] += 1
            FN[labels[i]] += 1

    return TP, FP, FN, ACC

def getResults(TP, FP, FN):
    n = TP.shape[0]
    P = torch.zeros(n)
    R = torch.zeros(n)
    F1 = torch.zeros(n)

    P = TP/(FP+TP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)

    for i in range(n):
        if TP[i]==0:
            P[i] = 0
            R[i] = 0
            F1[i] = 0

    return P, R, F1, F1.mean()