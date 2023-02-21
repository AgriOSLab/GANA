import os
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm

from utils import *
from Model import *
from Config import *

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

def predict():

    test_dataset = load_dataset(CFG.data_path+"test.txt")
    testSet = prepareDataset(test_dataset, CFG)
    test_loader = DataLoader(testSet,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             num_workers=1,
                             drop_last=False,
                             pin_memory=True)

    model = BertClassificationPool(CFG)
    model.to(device)

    result_file = open(CFG.output_dir+"demo.txt", 'w', encoding='utf8')

    if os.path.exists(CFG.output_dir+"/checkpoint.txt"):
        with open(CFG.output_dir+"/checkpoint.txt", 'r', encoding='utf8') as f:
            checkpoint = "model_10.pth"
            parameters = torch.load(CFG.output_dir + checkpoint)["model"]
            model.load_state_dict(parameters)
            LOGGER.info(f"Load from {checkpoint}")

    result_file.write(f"epoch: {checkpoint}\n")
    model.eval()
    with torch.no_grad():
        TP = torch.zeros(CFG.num_labels)
        FP = torch.zeros(CFG.num_labels)
        FN = torch.zeros(CFG.num_labels)
        ACC = 0
        N = 0
        for batch in tqdm(test_loader):
            input_ids = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            labels = batch[3].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)

            preds = torch.argmax(logits, dim=-1)

            N += len(labels)
            TP, FP, FN, ACC = cal(preds, labels, TP, FP, FN, ACC)
        P, R, F1, Macro_F = getResults(TP, FP, FN)

        LOGGER.info("test:\n acc: %f" % (ACC / N))
        result_file.write("test result\n")
        LOGGER.info("labels\tP\tR\tF1\t")

        for i in range(CFG.num_labels):
            LOGGER.info("%d\t%f\t%f\t%f\t" % (i, P[i], R[i], F1[i]))
            result_file.write("%d\t%f\t%f\t%f\t\n" % (i, P[i], R[i], F1[i]))
        LOGGER.info("average\t%f\t%f\t%f\t" % (P.mean(), R.mean(), Macro_F))
        result_file.write("average\t%f\t%f\t%f\t\n" % (P.mean(), R.mean(), Macro_F))


    result_file.close()
    return Macro_F


if __name__=="__main__":

    CFG = config()

    CFG.output_dir = 'outputs/Bert/'


    LOGGER = get_logger(CFG.output_dir + 'train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(CFG.seed)

    predict()