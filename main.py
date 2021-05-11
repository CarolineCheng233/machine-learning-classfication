import os.path as osp
import os
# from opts import parse
from opts import parse
from data import DataProcessPipeline, ItemDataset
from torch.utils.data import DataLoader
from model import BERT, MLP, Classifier
import torch.optim as optim
import torch.nn as nn


def train(args):
    data_pipeline = DataProcessPipeline(args.bert_path, args.allowed_keys)
    dataset = ItemDataset(args.file_name, data_pipeline, test_mode=False, allowed_keys=args.allowed_keys)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True)

    bert = BERT(pretrained=args.bert_path, freeze=True)
    mlp = MLP(layer_num=args.mlp_layer_num, dims=args.mlp_dims, with_bn=args.with_bn, act_type=args.act_type,
              last_w_bnact=args.last_w_bnact, last_w_softmax=args.last_w_softmax)
    model = Classifier(bert, mlp)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.sgd['lr'], momentum=args.sgd['momentum'])

    for i in range(args.epochs):
        iters = len(dataloader)
        for j, batch in enumerate(dataloader):
            # batch[0] = dict of tensors, batch[1] label tensor
            labels = batch[1]
            summaries = batch[0]['summary']
            optimizer.zero_grad()
            output = model(summaries)
            loss = loss_fn(output, labels)
            if j % args.log['iter'] == 0:
                print(f'Epoch: {i}, iter: {j} / {iters}')
            loss.backward()
            optimizer.step()


def eval(args):
    data_pipeline = DataProcessPipeline(args.bert_path, args.allowed_keys)
    dataset = ItemDataset(args.file_name, data_pipeline, test_mode=True, allowed_keys=args.allowed_keys)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=2, pin_memory=True)
    for i, batch in enumerate(dataloader):
        # batch dict of tensors
        pass
        # import pdb;
        # pdb.set_trace()


def main():
    args = parse()
    if args.mode == 'train':
        train(args)
    else:
        eval(args)


if __name__ == '__main__':
    main()
