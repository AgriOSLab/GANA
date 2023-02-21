import torch.nn as nn
from transformers import BertModel, AutoConfig
import torch
import random

class BertClassification(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = AutoConfig.from_pretrained(cfg.model_path+"/config")
        self.Bert = BertModel(config=self.config, add_pooling_layer=False).from_pretrained(cfg.model_path+"model")
        # self.activation = nn.Tanh()
        # self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(self.config.hidden_size, cfg.num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids, mode="train"):
        Bert_output = self.Bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

        logits = Bert_output["last_hidden_state"][:, 0, :]
        # logits = self.activation(logits)
        # logits = self.drop(logits)
        logits = self.fc(logits)

        if mode=="train":
            return logits
        else:
            return self.softmax(logits)
        # return self.softmax(logits)

class BertClassificationPool(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = AutoConfig.from_pretrained(cfg.model_path+"/config")
        self.Bert = BertModel(config=self.config, add_pooling_layer=False).from_pretrained(cfg.model_path+"model")
        self.pool = nn.MaxPool2d(kernel_size=[1,64], stride=[1, 64])
        self.flat = nn.Flatten()
        # self.activation = nn.Tanh()
        # self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(12*128, cfg.num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids, mode="train"):
        Bert_output = self.Bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

        logits = self.pool(Bert_output["last_hidden_state"])
        logits = self.flat(logits)
        # logits = self.activation(logits)
        # logits = self.drop(logits)
        logits = self.fc(logits)

        if mode=="train":
            return logits
        else:
            return self.softmax(logits)
        # return self.softmax(logits)

class BertBase(nn.Module):
    def __init__(self, cfg):
        super(BertBase, self).__init__()
        self.config = AutoConfig.from_pretrained(cfg.model_path + "/config")
        self.Bert = BertModel(config=self.config, add_pooling_layer=False).from_pretrained(cfg.model_path + "model")

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.Bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        return outputs["last_hidden_state"][:,0,:]


class GANBert(nn.Module):
    def __init__(self, CFG):
        super(GANBert, self).__init__()
        self.config = CFG
        self.num_fakelabels = len(CFG.fake_labels)
        self.num_labels = CFG.num_labels + 1
        self.fixGenerator = True

        self.backbone = BertBase(CFG)
        self.generator = Generator(CFG)
        self.discriminator = Discrimitor(CFG)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        if self.training:
            fakeSample, fakeLabels = self.generator()

            if self.fixGenerator:
                fakeLabels = (self.num_labels - 1) * torch.ones(self.num_fakelabels, dtype=torch.int).to(0)
                realSample = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                samples = torch.cat((fakeSample, realSample), dim=0)
                labels = torch.cat((fakeLabels, labels), dim=0)

                logits = self.discriminator(samples)

                return logits, labels
            else:
                logits = self.discriminator(fakeSample)
                return logits, fakeLabels
        else:
            realSample = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = self.discriminator(realSample)
            # logits = self.softmax(logits)
            return logits

class GANBert_for_TSNE(nn.Module):
    def __init__(self, CFG):
        super(GANBert_for_TSNE, self).__init__()
        self.config = CFG
        self.num_fakelabels = len(CFG.fake_labels)
        self.num_labels = CFG.num_labels + 1
        self.fixGenerator = True

        self.backbone = BertBase(CFG)
        self.generator = Generator(CFG)
        self.discriminator = Discrimitor(CFG)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, flag=True):
        if flag:
            realSample = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            return realSample
        else:
            generated_embedding, labels = self.generator()
            return generated_embedding, labels


class Generator(nn.Module):
    def __init__(self, CFG):
        super(Generator, self).__init__()
        self.labels = CFG.fake_labels
        self.input_dim = CFG.max_len
        self.hidden_dim = 768

        self.fc = nn.Linear(self.input_dim, self.hidden_dim)

    def generateFakeInputs(self):
        """
        generate the random inputs of Generator
        :return: size of [num_labels, input_dim]
        """

        noise = torch.normal(mean=0, size=(len(self.labels), self.input_dim), std=0.5)
        inputs = torch.ones(size=(len(self.labels), self.input_dim), dtype=torch.float32)
        inputs = torch.tensor(self.labels).reshape((len(self.labels), -1)) * inputs

        inputs += noise
        inputs = inputs.cuda(0)
        return inputs

    def forward(self):
        random.shuffle(self.labels)
        inputs = self.generateFakeInputs()

        embeddings = self.fc(inputs)

        labels = torch.tensor(self.labels).cuda(0)
        return embeddings, labels


class Discrimitor(nn.Module):
    def __init__(self, CFG):
        super(Discrimitor, self).__init__()
        self.hidden_dim = 768
        self.num_labels = CFG.num_labels + 1

        self.drop = nn.Dropout(CFG.dropout)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, batch):
        output = self.activation(batch)
        output = self.drop(output)
        output = self.fc(output)
        return output