from setfit import SetFitModel
from setfit import TrainingArguments
from setfit import Trainer
from utils.dataset import CustomDataset
from datasets import load_dataset
from utils.load_data import *
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.labels = [0,1,2]
    args = TrainingArguments(
        batch_size = 1,
        num_epochs = 10
    )

    train_dataset = load_dataset("./data")['train']

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        column_mapping={
            "text": "text",
            "stars": "label"
        }
    )
    print("yes")
    trainer.train()


if __name__ == '__main__':
    main()