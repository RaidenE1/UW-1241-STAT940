from setfit import (
    SetFitModel,
    TrainingArguments,
    Trainer,
    sample_dataset
)
from utils.dataset import CustomDataset
from datasets import load_dataset
from utils.load_data import *
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.labels = [1,2,3]
    batch_size = 80,10
    num_epochs = 20
    sample = 32
    body_learning_rate = 1e-4, 5e-5
    head_learning_rate = 5e-2
    args = TrainingArguments(
        output_dir = "./out", 
        batch_size = batch_size,
        num_epochs = num_epochs,
        sampling_strategy="oversampling",
        body_learning_rate=body_learning_rate,
        head_learning_rate=head_learning_rate
    )

    train_dataset = sample_dataset(load_dataset("./data")['train'], label_column="stars", num_samples=sample)

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
    df.to_csv('./result_s%d_b%d_e%d.csv' % (sample, batch_size, num_epochs), index=False)

if __name__ == '__main__':
    main()  