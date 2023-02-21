from transformers import AutoTokenizer

class config():
    def __init__(self):
        self.batch_size = 32
        self.data_path = "./data/"
        self.dropout = 0.5
        self.epochs = 20
        self.fake_labels = [0, 2, 6]*10
        self.isFromCheckPoint = False
        self.k_Discriminator = 20
        self.lr = 5e-6
        self.lr_G = 5e-6
        self.max_len = 128
        self.model_path = "./models/bert-base-chinese/"
        self.num_labels = 9
        self.output_dir = './outputs/BertPool/'
        self.seed = 100
        self.tokenizer = self.get_Tokenizer()

        self.isWeighted = True
        # self.get_vocab_size()

    def __str__(self):
        properties = self.__dict__
        attr = {}
        for property in properties:
           attr[property] = self.__getattribute__(property)

        self.str = ""
        for key in attr.keys():
            self.str += str(key)+": "+str(attr[key])+"\n"
        return self.str

    def get_Tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_path+"/tokenizer")
