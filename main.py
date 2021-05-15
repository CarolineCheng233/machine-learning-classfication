from opts import parse
from data import DataProcessPipeline, ItemDataset
from torch.utils.data import DataLoader, DistributedSampler
from model import BERT, MLP, Classifier
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np

from mmcv import ProgressBar

from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def train(rank, args):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:65534',
                            rank=rank, world_size=args.GPUS)
    torch.cuda.set_device(rank)
    data_pipeline = DataProcessPipeline(args.bert_path, args.allowed_keys)
    dataset = ItemDataset(args.train_file, data_pipeline, test_mode=False, allowed_keys=args.allowed_keys)
    dataset_sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=dataset_sampler, shuffle=True, batch_size=args.batch_size, pin_memory=True)

    bert = BERT(pretrained=args.bert_path, freeze=True)
    mlp = MLP(layer_num=args.mlp_layer_num, dims=args.mlp_dims, with_bn=args.with_bn, act_type=args.act_type,
              last_w_bnact=args.last_w_bnact, last_w_softmax=args.last_w_softmax)
    model = Classifier(bert, mlp)
    model.cuda(rank)
    model = DDP(model, device_ids=[rank])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.sgd['lr'], momentum=args.sgd['momentum'])

    best = 0
    if rank == 0:
        epoch_pb = ProgressBar(args.epochs)
        epoch_pb.start()
        writer = SummaryWriter(args.log_path)
    for i in range(args.epochs):
        model.train()
        dataset_sampler.set_epoch(i)
        iters = len(dataloader)
        if rank == 0:
            iter_pb = ProgressBar(iters)
            iter_pb.start()
        for j, batch in enumerate(dataloader):
            labels = batch[1].cuda()
            summaries = batch[0]['summary'].cuda()
            optimizer.zero_grad()
            output = model(summaries)
            loss = loss_fn(output, labels)
            if rank == 0 and j % args.log['iter'] == 0:
                print(f'Epoch: {i}, iter: {j} / {iters}, loss: {loss}')
            if rank == 0:
                writer.add_scalar('loss', loss, global_step=i * iters + j + 1)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=i * iters + j + 1)
            loss.backward()
            optimizer.step()
            if rank == 0:
                iter_pb.update()
        if rank == 0:
            epoch_pb.update()

        dist.barrier()
        if rank == 0:
            accuracy = val(model, data_pipeline, args)
            writer.add_scalar('accuracy', accuracy, global_step=(i + 1) * iters)
            if accuracy > best:
                print(f'save best model at epoch {i}')
                torch.save(model.state_dict(), args.ckpt_path)
                best = accuracy


def val(model, data_pipeline, args):
    dataset = ItemDataset(args.val_file, data_pipeline, test_mode=True, allowed_keys=args.allowed_keys)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, drop_last=False)

    model.eval()

    results = []
    labels = []
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            label = batch[1].detach().numpy().reshape(-1, 1)
            summary = batch[0]['summary']
            output = model(summary).detach().numpy().argmax(1).reshape(-1, 1)
            results.append(output)
            labels.append(label)
    labels = np.concatenate(labels, axis=1)
    results = np.concatenate(results, axis=1)
    accuracy = sum(labels == results) / len(labels)
    return accuracy


def test(args):
    data_pipeline = DataProcessPipeline(args.bert_path, args.allowed_keys)
    dataset = ItemDataset(args.test_file, data_pipeline, test_mode=False, allowed_keys=args.allowed_keys)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, drop_last=False)

    bert = BERT(pretrained=args.bert_path, freeze=True)
    mlp = MLP(layer_num=args.mlp_layer_num, dims=args.mlp_dims, with_bn=args.with_bn, act_type=args.act_type,
              last_w_bnact=args.last_w_bnact, last_w_softmax=args.last_w_softmax)
    model = Classifier(bert, mlp)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    results = []
    labels = []
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            label = batch[1].detach().numpy().reshape(-1, 1)
            summary = batch[0]['summary']
            output = model(summary).detach().numpy().argmax(1).reshape(-1, 1)
            results.append(output)
            labels.append(label)
    labels = np.concatenate(labels, axis=1)
    results = np.concatenate(results, axis=1)
    accuracy = sum(labels == results) / len(labels)
    return accuracy


def main():
    args = parse()
    if args.mode == 'train':
        mp.spawn(train, nprocs=args.GPUS, args=(args,))
    else:
        test(args)


if __name__ == '__main__':
    main()
