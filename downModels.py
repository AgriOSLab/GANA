from transformers import AutoModel, AutoConfig, AutoTokenizer


name = "bert-base-chinese"
model = AutoModel.from_pretrained(name)
config = AutoConfig.from_pretrained(name)
config.output_hidden_states = True
tokenizer = AutoTokenizer.from_pretrained(name)

model.save_pretrained("models/"+name+"/model/")
config.save_pretrained("models/"+name+"/config/")

tokenizer.save_pretrained("models/"+name+"/tokenizer/")