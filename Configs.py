from transformers import AutoTokenizer
import torch

class config():
    def __init__(self):
        self.batch_size = 32
        self.data_path = "./data/"
        self.dropout = 0.5
        self.embedding_path = self.data_path+"/pretrained.pth"
        self.embedding_pretrained = False
        self.epochs = 20
        self.fake_labels = [0, 2, 6]
        self.hidden_size = 768

        self.isWeighted = False
        self.isFromCheckPoint = True
        self.lr = 1e-5
        self.lr_G = 5e-6
        self.max_len = 128
        self.num_labels = 9
        self.num_features = 50
        self.k_Discriminator = 30
        self.kernels = [3, 5, 7]
        self.output_dir = 'outputs/GAN_textcnn-k30-unbalance1/'
        self.seed = 100
        self.tokenizer = self.get_Tokenizer()
        self.vocab_size = 21128

        self.weights = [100,8,100,8,1,10,1000,1,12]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.get_vocab_size()

    def __str__(self):
        properties = self.__dict__
        attr = {}
        for property in properties:
            attr[property] = self.__getattribute__(property)

        self.str = ""
        for key in attr.keys():
            self.str += str(key) + ": " + str(attr[key]) + "\n"
        return self.str

    def get_vocab_size(self):
        size = 0
        with open(self.data_path+"vocab.txt",'r',encoding='utf8') as f:
            for line in f:
                size += 1
        self.vocab_size = size

    def get_Tokenizer(self):
        return AutoTokenizer.from_pretrained(self.data_path+"/tokenizer")
