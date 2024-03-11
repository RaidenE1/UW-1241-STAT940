from transformers import (
    Trainer,
    TrainingArguments
)
import torch
from tqdm import tqdm
from utils.dataset import CustomDataset
from datasets import load_dataset
from utils.load_data import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import pipeline

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

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    # test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    for epoch in n
    
    test_data = load_test_labels("./test.csv")
    IDs = list(test_data.keys())
    IDs.sort()
    texts = [
        test_data[ID] for ID in IDs
    ]
    predicts = []
    for text in tqdm(texts):
        token = tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        token.to(torch.device('cuda'))
        output = model(**token)
        _, predict = output.logits.max(1)
        predict = int(predict.to(torch.device('cpu')))
        predicts.append(predict + 1)
    df = pd.DataFrame({'ID': IDs, 'stars': predicts})
    df.to_csv('./out/result_xlnet_base.csv', index=False)

if __name__ == '__main__':
    main()  