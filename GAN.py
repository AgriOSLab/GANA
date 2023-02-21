import os
import gc
import numpy as np
import torch.nn as nn
import torch
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import *
from Config import *
from Model import *

def get_logger(filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_everything(seed=42):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_eval():
    trainSets = load_dataset(CFG.data_path+"train.txt")
    trainSets = prepareDataset(trainSets, CFG)
    train_loader = DataLoader(trainSets,
                              batch_size=CFG.batch_size-len(CFG.fake_labels),
                              shuffle=True,
                              num_workers=5,
                              drop_last=False,
                              pin_memory=True)

    devSets = load_dataset(CFG.data_path + "dev.txt")
    devSets = prepareDataset(devSets, CFG)
    dev_loader = DataLoader(devSets,
                            batch_size=CFG.batch_size,
                            shuffle=True,
                            num_workers=5,
                            drop_last=False,
                            pin_memory=True)

    testSets = load_dataset(CFG.data_path + "test.txt")
    testSets = prepareDataset(testSets, CFG)
    test_loader = DataLoader(testSets,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             num_workers=5,
                             drop_last=False,
                             pin_memory=True)

    model = GANBert(CFG)

    model_struct = open(CFG.output_dir+"model_struct.txt",'w',encoding='utf8')
    model_struct.write(str(model))
    model_struct.close()

    backbone_optimizer = torch.optim.Adam(model.backbone.parameters(), lr=CFG.lr)
    D_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=CFG.lr)
    G_optimizer = torch.optim.Adam(model.generator.parameters(), lr=CFG.lr_G)

    # loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([20,1,30,10,1,3,30,1,8])).float()).cuda(0)
    loss_function = nn.CrossEntropyLoss().cuda(device)
    model.to(device)

    start_epoch = -1

    if CFG.isFromCheckPoint:
        if os.path.exists(CFG.output_dir + "/checkpoint.txt"):
            with open(CFG.output_dir + "/checkpoint.txt", 'r', encoding='utf8') as f:
                checkpoint = f.readline().strip()
                parameters = torch.load(CFG.output_dir + checkpoint)["model"]
                model.load_state_dict(parameters)
                LOGGER.info(f"Load from {checkpoint}")
                start_epoch = checkpoint.replace("model_", "").replace(".pth", "")
                start_epoch = int(start_epoch)
        result_file = open(CFG.output_dir + "result.txt", 'a', encoding='utf8')
    else:
        result_file = open(CFG.output_dir+"result.txt",'w',encoding='utf8')

    best_score = 0.0
    for epoch in range(start_epoch+1, CFG.epochs):
        train_model(epoch, model, train_loader, backbone_optimizer, D_optimizer, G_optimizer, loss_function)

        LOGGER.info(f"save model of epoch: {epoch}")
        torch.save({'model': model.state_dict()},
                   CFG.output_dir+f"model_{epoch}.pth")

        score = eval_model(epoch, model, dev_loader, result_file)

        if score > best_score:
            best_score = score
            with open(CFG.output_dir+"checkpoint.txt", 'w', encoding='utf8') as f:
                f.write(f"model_{epoch}.pth")

    eval_model(0, model, test_loader, result_file, mode="test")
    result_file.close()


def train_model(epoch, model, train_loader, backbone_optimizer, D_optimizer, G_optimizer, loss_function):
    model.train()
    backboneLoss = 0.0
    GLoss = 0.0

    with torch.enable_grad():
        for batch in tqdm(train_loader):
            model.fixGenerator = True
            input_ids = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            labels = batch[3].to(device)

            logits, labels = model(input_ids, attention_mask, token_type_ids, labels)

            ls = loss_function(logits, labels)
            model.generator.requires_grad_(False)
            model.backbone.requires_grad_(True)
            model.discriminator.requires_grad_(True)
            G_optimizer.zero_grad()
            backbone_optimizer.zero_grad()
            D_optimizer.zero_grad()
            ls.backward()
            D_optimizer.step()
            backbone_optimizer.step()

            for i in range(CFG.k_Discriminator):
                ## fix Discriminator
                model.fixGenerator = False
                logits, labels = model()
                """
                G_loss --> Bert, G
                """
                G_loss = loss_function(logits, labels)
                model.discriminator.requires_grad_(False)
                model.generator.requires_grad_(True)
                model.backbone.requires_grad_(True)
                G_optimizer.zero_grad()
                backbone_optimizer.zero_grad()
                D_optimizer.zero_grad()
                G_loss.backward()
                backbone_optimizer.step()
                G_optimizer.step()

            backboneLoss += ls.item()
            GLoss += G_loss.item()

        LOGGER.info(f"Epoch: {epoch}, backbone loss: {backboneLoss / len(train_loader)}")
        LOGGER.info(f"Epoch: {epoch}, Generator loss: {GLoss / len(train_loader)}")

def eval_model(epoch, model, loader, result_file, mode="dev"):
    model.eval()
    with torch.no_grad():
        TP = torch.zeros(CFG.num_labels)
        FP = torch.zeros(CFG.num_labels)
        FN = torch.zeros(CFG.num_labels)
        ACC = 0
        N = 0
        for batch in tqdm(loader):
            input_ids = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            labels = batch[3].to(device)

            preds = model(input_ids, attention_mask, token_type_ids)

            preds = torch.argmax(preds, dim=-1)

            N += len(labels)
            TP, FP, FN, ACC = cal(preds, labels, TP, FP, FN, ACC)
        P, R, F1, Macro_F = getResults(TP, FP, FN)
        if mode == "dev":
            LOGGER.info("dev:\n acc: %f" % (ACC / N))
            result_file.write("epoch: %d\n" % epoch)
        else:
            LOGGER.info("test:\n acc: %f" % (ACC / N))
            result_file.write("test result\n")
        LOGGER.info("labels\tP\tR\tF1\t")
        for i in range(CFG.num_labels):
            LOGGER.info("%d\t%f\t%f\t%f\t" % (i, P[i], R[i], F1[i]))
            result_file.write("%d\t%f\t%f\t%f\t\n" % (i, P[i], R[i], F1[i]))
        LOGGER.info("average\t%f\t%f\t%f\t" % (P.mean(), R.mean(), Macro_F))
        result_file.write("average\t%f\t%f\t%f\t\n" % (P.mean(), R.mean(), Macro_F))

        return Macro_F


if __name__=='__main__':
    kA = [60, 70, 80, 90, 100]

    CFG = config()
    for k in kA:
        CFG.k_Discriminator = k
        CFG.output_dir = 'outputs/GAN-BERT-' + "k" + str(CFG.k_Discriminator) + '/'

        if not os.path.isdir(CFG.output_dir):
            os.mkdir(CFG.output_dir)

        LOGGER = get_logger(CFG.output_dir + 'train')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed_everything(CFG.seed)

        with open(CFG.output_dir+"config.txt", 'w', encoding='utf8') as f:
            f.write(CFG.__str__())

        train_eval()
        torch.cuda.empty_cache()
        gc.collect()