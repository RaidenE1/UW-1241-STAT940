import pandas as pd

def load_data(path = './data/train.csv'):
    df = pd.read_csv(path)
    data = {}
    for row in df.itertuples():
        data[row.text] = int(row.stars) - 1
    return data

def load_test_labels(path = './data/test.csv'):
    df = pd.read_csv(path)
    data = {}
    for row in df.itertuples():
        data[int(row.ID)] = row.text
    return data

if __name__ == '__main__':
    print(load_data())
