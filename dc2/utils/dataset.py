from torch.utils.data import Dataset
from transformers import BertTokenizer
from .load_data import load_data
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, data):
        self.label = load_data()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.texts = data

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.tokenizer(
                self.texts[idx],
                padding = "max_length",
                max_length = 512,
                truncation = True,
                return_tensors = "pt"
            )
        label = self.label[self.texts[idx]]
        return text, label


class TestDataset(Dataset):
    def __init__(self, labels):
        self.IDs = list(labels.keys())
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.texts = [
            self.tokenizer(
                labels[ID],
                padding = "max_length",
                max_length = 512,
                truncation = True,
                return_tensors = "pt"
            ) for ID in self.IDs
        ]
    
    def __len__(self):
        return len(self.IDs)
    
    def __getitem__(self, idx):
        return self.IDs[idx], self.texts[idx]