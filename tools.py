import argparse
import pandas as pd
import numpy as np


def val_split():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--number', type=int, default=7544)
    parser.add_argument('--val', type=str, required=True)
    args = parser.parse_args()
    test, val, label, number = args.test, args.val, args.label, args.number
    test = pd.read_csv(test)
    data = test.values
    total = len(data)
    with open(label, 'r', encoding='utf-8') as f:
        label = np.array([line.strip() for line in f])
    choices = np.random.choice(total, number, replace=False)
    val_data = data[choices]
    val_label = label[choices].reshape(number, 1)
    data = np.concatenate((val_data, val_label), axis=1)
    keys = list(test.keys().values)
    keys.append('fit')
    keys = np.array(keys)
    df = pd.DataFrame(data, columns=pd.Index(keys))
    df.to_csv(val, index=False)


def read_val():
    val = "val.txt"
    data = pd.read_csv(val)
    print(data.keys())
    print(data.values.shape)


def train_split():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--number', type=int, default=10000)
    parser.add_argument('--train_split', type=str, required=True)
    args = parser.parse_args()
    train, train_split, number = args.train, args.train_split, args.number
    train = pd.read_csv(train)
    data = train.values
    total = len(data)
    choices = np.random.choice(total, number, replace=False)
    data = data[choices]
    df = pd.DataFrame(data, columns=train.keys())
    df.to_csv(train_split, index=False)


if __name__ == '__main__':
    train_split()