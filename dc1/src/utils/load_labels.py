import pandas as pd

# load train_labels.csv to a dict
def load_labels():
    label_dict = {}
    df = pd.read_csv("data/train_labels.csv")
    for row in df.itertuples():
        if row.id not in label_dict:
            label_dict[row.id] = row.label
        else:
            raise ValueError("Duplicate")
    return label_dict


if __name__ == "__main__":
    load_labels()