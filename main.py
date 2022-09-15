import os
import gc
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm

from utils import *
from Models import *
from Configs import *

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
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_eval():

    train_dataset = load_dataset(CFG.data_path+"train.txt")
    trainSet = prepareDataset(train_dataset, CFG)
    train_loader = DataLoader(trainSet,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=5,
                              drop_last=False,
                              pin_memory=True)

    dev_dataset = load_dataset(CFG.data_path+"dev.txt")
    devSet = prepareDataset(dev_dataset, CFG)
    dev_loader = DataLoader(devSet,
                            batch_size=CFG.batch_size,
                            shuffle=False,
                            num_workers=5,
                            pin_memory=True,
                            drop_last=False)

    test_dataset = load_dataset(CFG.data_path+"test.txt")
    testSet = prepareDataset(test_dataset, CFG)
    test_loader = DataLoader(testSet,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             num_workers=5,
                             drop_last=False,
                             pin_memory=True)

    model = TextCNN(CFG)

    model_struct = open(CFG.output_dir + "model_struct.txt", 'w', encoding='utf8')
    model_struct.write(str(model))
    model_struct.close()
    model.to(device)

    optimizer = Adam(params=model.parameters(), lr=CFG.lr)
    if CFG.isWeighted:
        loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(CFG.weights)).float()).cuda(device)
    else:
        loss_function = nn.CrossEntropyLoss().cuda(device)

    start_epoch = -1

    if CFG.isFromCheckPoint:
        if os.path.exists(CFG.output_dir+"/checkpoint.txt"):
            with open(CFG.output_dir+"/checkpoint.txt", 'r', encoding='utf8') as f:
                checkpoint = f.readline().strip()
                parameters = torch.load(CFG.output_dir + checkpoint)["model"]
                model.load_state_dict(parameters)
                LOGGER.info(f"Load from {checkpoint}")
                start_epoch = checkpoint.replace("model_", "").replace(".pth","")
                start_epoch = int(start_epoch)
        result_file = open(CFG.output_dir + "result.txt", 'a', encoding='utf8')
    else:
        result_file = open(CFG.output_dir + "result.txt", 'w', encoding='utf8')
    best_score = 0.0
    for epoch in range(start_epoch+1, CFG.epochs):
        train(epoch, model, train_loader, optimizer, loss_function)
        score = eval(epoch, model, dev_loader, result_file)
        if score>best_score:
            best_score = score
            with open(CFG.output_dir+"checkpoint.txt", 'w', encoding='utf8') as f:
                f.write(f"model_{epoch}.pth")
        LOGGER.info(f"save model of epoch: {epoch}")
        torch.save({'model': model.state_dict()},
                   CFG.output_dir + f"model_{epoch}.pth")
    eval(0, model, test_loader, result_file, "test")
    result_file.close()

def train(epoch, model, train_loader, optimizer, loss_function):
    model.train()
    train_loss = 0

    with torch.enable_grad():
        for batch in tqdm(train_loader):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)

            preds = model(input_ids)

            loss = loss_function(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
    optimizer.zero_grad()
    LOGGER.info(f"Epoch: {epoch}, loss: {train_loss/len(train_loader)}")

def eval(epoch, model, loader, result_file, mode="dev"):
    model.eval()
    with torch.no_grad():
        TP = torch.zeros(CFG.num_labels)
        FP = torch.zeros(CFG.num_labels)
        FN = torch.zeros(CFG.num_labels)
        ACC = 0
        N = 0
        for batch in tqdm(loader):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)

            preds = model(input_ids)

            preds = torch.argmax(preds, dim=-1)

            N += len(labels)
            TP, FP, FN, ACC = cal(preds, labels, TP, FP, FN, ACC)
        P, R, F1, Macro_F = getResults(TP, FP, FN)
        if mode=="dev":
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


if __name__=="__main__":
    kA = [40,50,60,70,80,90,100]

    CFG = config()

    for k in kA:
        CFG.k_Discriminator = k
        CFG.output_dir = 'outputs/GAN_textcnn-'+"k"+str(CFG.k_Discriminator)+'-unbalance1/'
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