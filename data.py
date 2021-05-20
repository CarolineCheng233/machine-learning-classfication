import numpy as np
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


def read():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    args = parser.parse_args()
    train_file = args.train_file
    data = pd.read_csv(train_file)
    # data.keys() = ['age', 'body type', 'bust size', 'category', 'height', 'item_id',
    #                'rating', 'rented for', 'review_date', 'review_summary', 'review_text',
    #                'size', 'user_id', 'weight', 'fit']
    # fit: large, fit, small


class DataProcessPipeline:

    def __init__(self, tokenizer_dir, allowed_keys=["body type"]):
        super().__init__()
        if 'fit' in allowed_keys:
            allowed_keys.remove('fit')
        self.allowed_keys = allowed_keys
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)

    def __call__(self, data):
        result = {}
        if "review_summary" in data:
            try:
                result['summary'] = self.tokenizer([data["review_summary"]], truncation=True,
                                                   padding="max_length", return_tensors="pt")
            except:
                data["review_summary"] = 'nan'
                result['summary'] = self.tokenizer([data["review_summary"]], truncation=True,
                                                   padding="max_length", return_tensors="pt")
            for key in result['summary']:
                result['summary'][key] = result['summary'][key].reshape((-1,) + result['summary'][key].shape[2:])
        return result


class ItemDataset(Dataset):
    def __init__(self, file_name, pipeline, test_mode=False, allowed_keys=["body type"]):
        if 'fit' in allowed_keys:
            allowed_keys.remove('fit')
        self.file_name = file_name
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.allowed_keys = allowed_keys
        self.data = pd.read_csv(file_name)
        self.data_by_keys = dict()
        for key in allowed_keys:
            self.data_by_keys[key] = self.data[key].values
        self.label2idx = dict(large=0, fit=1, small=2)
        self.idx2label = {0: 'large', 1: 'fit', 2: 'small'}
        self.labels = self.data['fit'].values
        self.one_hot_labels = []
        for label in self.labels:
            self.one_hot_labels.append(self.label2idx[label])
        self.one_hot_labels = np.array(self.one_hot_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result = dict()
        for key in self.allowed_keys:
            result[key] = self.data_by_keys[key][idx]
        return self.pipeline(result), self.one_hot_labels[idx]


class ItemDataLoader(DataLoader):
    pass


# if __name__ == '__main__':
#     allowd_keys = ["review_summary"]
#     pipeline = DataProcessPipeline(allowd_keys)
#     dataset = ItemDataset()
