import os.path as osp
import os
# from opts import parse
from opts import parse
from data import DataProcessPipeline, ItemDataset
from torch.utils.data import DataLoader


def train(args):
    data_pipeline = DataProcessPipeline(args.bert_path, args.allowed_keys)
    dataset = ItemDataset(args.file_name, data_pipeline, test_mode=False, allowed_keys=args.allowed_keys)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=2, pin_memory=True)
    for i, batch in dataloader:
        import pdb; pdb.set_trace()


def eval(args):
    pass


def main():
    args = parse()
    if args.mode == 'train':
        train(args)
    else:
        eval(args)


if __name__ == '__main__':
    main()
