from torch.utils.data import Dataset
import torch

def read_vocab(vocab_path):
    """
    读取词汇表，构建 词汇-->ID 映射字典
    :param vocab_path: 词表文件路径
    :return: 词表，word_to_id
    """
    words = [word.replace('\n', '').strip() for word in open(vocab_path, encoding='UTF-8')]
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def load_dataset(path):
    dataset = []

    with open(path,'r',encoding='utf8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            if len(line)==2:
                dataset.append({"sentence":line[0], "label":int(line[1])})
    return dataset

class prepareDataset(Dataset):
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        input = self.prepare_inputs(self.cfg, self.dataset[item]["sentence"])
        label = self.prepare_labels(self.dataset[item]["label"])

        return input, label

    def prepare_inputs(self, CFG, text):
        input_ids = []
        input_ids.append(CFG.tokenizer.convert_tokens_to_ids('[CLS]'))

        for c in text:
            input_ids.append(CFG.tokenizer.convert_tokens_to_ids(c))

        input_ids.append(CFG.tokenizer.convert_tokens_to_ids('[SEP]'))
        if len(input_ids) > CFG.max_len:
            input_ids = input_ids[0:CFG.max_len]
        else:
            while len(input_ids) < CFG.max_len:
                input_ids.append(CFG.tokenizer.convert_tokens_to_ids('[PAD]'))

        return torch.tensor(input_ids, dtype=torch.long)

    def prepare_labels(self, label):
        return torch.tensor(label, dtype=torch.long)

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