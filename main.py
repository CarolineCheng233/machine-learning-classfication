from opts import parse
from data import DataProcessPipeline, ItemDataset
from torch.utils.data import DataLoader
from model import BERT, MLP, Classifier
import torch.optim as optim
import torch.nn as nn
import torch

import os
import os.path as osp
import numpy as np

from mmcv import ProgressBar

from torch.utils.tensorboard import SummaryWriter


# def init_slurm(args):
#     proc_id = int(os.environ['SLURM_PROCID'])
#     ntasks = int(os.environ['SLURM_NTASKS'])
#     node_list = os.environ['SLURM_NODELIST']
#     num_gpus = torch.cuda.device_count()
#     torch.cuda.set_device(proc_id % num_gpus)
#     addr = subprocess.getoutput(
#         f'scontrol show hostname {node_list} | head -n1')
#     # specify master port
#     if args.port is not None:
#         os.environ['MASTER_PORT'] = str(args.port)
#     elif 'MASTER_PORT' in os.environ:
#         pass  # use MASTER_PORT in the environment variable
#     else:
#         # 29500 is torch.distributed default port
#         os.environ['MASTER_PORT'] = '29500'
#     os.environ['MASTER_ADDR'] = addr
#     os.environ['WORLD_SIZE'] = str(ntasks)
#     os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
#     os.environ['RANK'] = str(proc_id)


def train(args):
    data_pipeline = DataProcessPipeline(args.bert_path, args.allowed_keys)
    dataset = ItemDataset(args.train_file, data_pipeline, test_mode=False, allowed_keys=args.allowed_keys)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=True)

    bert = BERT(pretrained=args.bert_path, freeze=args.bert_freeze)
    mlp = MLP(layer_num=args.mlp_layer_num, dims=args.mlp_dims, with_bn=args.with_bn, act_type=args.act_type,
              last_w_bnact=args.last_w_bnact, last_w_softmax=args.last_w_softmax)
    model = Classifier(bert, mlp).cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.sgd['lr'], momentum=args.sgd['momentum'])

    best = 0
    epoch_pb = ProgressBar(args.epochs)
    epoch_pb.start()
    writer = SummaryWriter(args.log_path)
    for i in range(args.epochs):
        model.train()
        iters = len(dataloader)
        iter_pb = ProgressBar(iters)
        iter_pb.start()
        for j, batch in enumerate(dataloader):
            labels = batch[1].cuda()
            summaries = batch[0]['summary']
            for key in summaries:
                summaries[key] = summaries[key].cuda()
            optimizer.zero_grad()
            output = model(summaries)
            loss = loss_fn(output, labels)
            if j % args.log['iter'] == 0:
                print(f'Epoch: {i}, iter: {j} / {iters}, loss: {loss}')
            writer.add_scalar('loss', loss, global_step=i * iters + j + 1)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=i * iters + j + 1)
            loss.backward()
            optimizer.step()
            iter_pb.update()
        epoch_pb.update()

        accuracy = val(model, data_pipeline, args)
        writer.add_scalar('accuracy', accuracy, global_step=(i + 1) * iters)
        print(f'accuracy at epoch: {i + 1} is {accuracy}')
        if accuracy > best:
            best = accuracy
            if args.save_model:
                print(f'save best model at epoch {i}')
                if not osp.exists(args.ckpt_dir):
                    os.makedirs(args.ckpt_dir)
                torch.save(model.state_dict(), osp.join(args.ckpt_dir, args.ckpt_name))


def val(model, data_pipeline, args):
    dataset = ItemDataset(args.val_file, data_pipeline, test_mode=True, allowed_keys=args.allowed_keys)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, drop_last=False)

    model.eval()

    results = []
    labels = []
    val_pb = ProgressBar(len(dataloader))
    val_pb.start()
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            label = batch[1].detach().numpy()
            summaries = batch[0]['summary']
            for key in summaries:
                summaries[key] = summaries[key].cuda()
            output = model(summaries).detach().cpu().numpy().argmax(1)
            results.append(output)
            labels.append(label)
            val_pb.update()
    labels = np.concatenate(labels, axis=0)
    results = np.concatenate(results, axis=0)
    accuracy = sum(labels == results) / len(labels)
    return accuracy


def test(args):
    data_pipeline = DataProcessPipeline(args.bert_path, args.allowed_keys)
    dataset = ItemDataset(args.test_file, data_pipeline, test_mode=False, allowed_keys=args.allowed_keys)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, drop_last=False)

    bert = BERT(pretrained=args.bert_path, freeze=True)
    mlp = MLP(layer_num=args.mlp_layer_num, dims=args.mlp_dims, with_bn=args.with_bn, act_type=args.act_type,
              last_w_bnact=args.last_w_bnact, last_w_softmax=args.last_w_softmax)
    model = Classifier(bert, mlp).cuda()
    model.load_state_dict(torch.load(osp.join(args.ckpt_dir, args.ckpt_path)))
    model.eval()

    results = []
    labels = []
    test_pb = ProgressBar(len(dataloader))
    test_pb.start()
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            label = batch[1].detach().numpy()
            summaries = batch[0]['summary']
            for key in summaries:
                summaries[key] = summaries[key].cuda()
            output = model(summaries).detach().cpu().numpy().argmax(1)
            results.append(output)
            labels.append(label)
            test_pb.update()
    labels = np.concatenate(labels, axis=0)
    results = np.concatenate(results, axis=0)
    accuracy = sum(labels == results) / len(labels)
    return accuracy


def main():
    args = parse()
    if args.mode == 'train':
        train(args)
    else:
        test(args)


if __name__ == '__main__':
    main()
