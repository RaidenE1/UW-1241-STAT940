from setfit import SetFitModel
# from setfit import TrainingArguments
# from setfit import Trainer
from transformers import (
    Trainer,
    TrainingArguments
)
from utils.dataset import CustomDataset
from datasets import load_dataset
from utils.load_data import *
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    data = load_data()
    texts = list(data.keys())
    labels = [data[text] for text in texts]
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    # test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='./out',          # output directory
        num_train_epochs=5,              # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=50,                # number of warmup steps for learning rate scheduler
        weight_decay=1e-5,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )
    model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.labels = [1,2,3]

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()
    
    test_data = load_test_labels("./test.csv")
    IDs = list(test_data.keys())
    IDs.sort()
    texts = [
        test_data[ID] for ID in IDs
    ]
    predicts = model.predict(texts)
    df = pd.DataFrame({'ID': IDs, 'stars': predicts})
    df.to_csv('./out/result.csv', index=False)

if __name__ == '__main__':
    main()