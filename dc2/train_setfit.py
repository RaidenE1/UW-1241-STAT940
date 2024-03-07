from setfit import SetFitModel
from setfit import TrainingArguments
from setfit import Trainer
from utils.dataset import CustomDataset
from datasets import load_dataset
from utils.load_data import *
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.labels = [1,2,3]
    args = TrainingArguments(
        batch_size = 64,
        num_epochs = 5,
        sampling_strategy="undersampling"
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