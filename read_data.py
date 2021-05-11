import numpy as np
import pandas as pd
import argparse
from torch.utils.data import Dataset


def read():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    args = parser.parse_args()
    train_file = args.train_file
    data = pd.read_csv(train_file)
    # import pdb; pdb.set_trace()
    # data.keys() = ['age', 'body type', 'bust size', 'category', 'height', 'item_id',
    #                'rating', 'rented for', 'review_date', 'review_summary', 'review_text',
    #                'size', 'user_id', 'weight', 'fit']
    # fit: large, fit, small


class DataProcessPipeline:

    def __init__(self):
        super().__init__()

    def __call__(self, results):
        return None


class ItemDataset(Dataset):
    def __init__(self, file_name, pipeline, batch_size, test_mode=False, allowed_keys=["body type"]):
        self.file_name = file_name
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.test_mode = test_mode
        self.allowed_keys = allowed_keys
        self.data = pd.read_csv(file_name)
        self.data_by_keys = dict()
        for key in allowed_keys:
            self.data_by_keys[key] = self.data[key].values
        if not test_mode:
            self.labels = self.data['fit'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        results = dict()
        for key in self.allowed_keys:
            results[key] = self.data_by_keys[key][idx]
        # return self.pipeline(results)
        if self.test_mode:
            return self.pipeline(results)



if __name__ == '__main__':
    read()
