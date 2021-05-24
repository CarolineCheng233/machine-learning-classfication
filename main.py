from opts import parse
from data import DataProcessPipeline, ItemDataset
from torch.utils.data import DataLoader
from model import BERT, MLP, Classifier
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

import os
import os.path as osp
import numpy as np

from mmcv import ProgressBar

from torch.utils.tensorboard import SummaryWriter
from pytorch_pretrained_bert import BertForSequenceClassification
from sklearn.metrics import f1_score


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


# def focal_loss(logits, labels, gamma):
#     output = F.log_softmax(logits, dim=1)
#     weight = torch.exp(output)
#     weighted_output = (1 - weight) ** gamma * output
#     loss = F.nll_loss(weighted_output, labels)
#     return loss


def train(args):
    data_pipeline = DataProcessPipeline(args.bert_path, args.allowed_keys)
    dataset = ItemDataset(args.train_file, data_pipeline, with_label=True, allowed_keys=args.allowed_keys)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=True, drop_last=False)

    bert = BERT(pretrained=args.bert_path, freeze=args.bert_freeze)
    # bert = BertForSequenceClassification.from_pretrained(
    #     'bert-base-multilingual-cased', num_labels=3)
    mlp = MLP(layer_num=args.mlp_layer_num, dims=args.mlp_dims, with_bn=args.with_bn, act_type=args.act_type,
              last_w_bnact=args.last_w_bnact, last_w_softmax=args.last_w_softmax)
    # mlp = None
    model = Classifier(bert, mlp).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.sgd['lr'], momentum=args.sgd['momentum'],
                          weight_decay=args.sgd['weight_decay'])
    # ratio = torch.tensor(args.ratio).cuda()

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
            optimizer.zero_grad()

            data, label = batch
            label = label.cuda()
            for key in data['text']:
                data['text'][key] = data['text'][key].squeeze().cuda()
            output = model(data)

            loss = F.cross_entropy(output, label)
            if j % args.log['iter'] == 0:
                print(f'Epoch: {i}, iter: {j} / {iters}, loss: {loss}')
            writer.add_scalar('loss', loss, global_step=i * iters + j + 1)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=i * iters + j + 1)
            loss.backward()
            optimizer.step()
            iter_pb.update()
        epoch_pb.update()

        score = val(model, data_pipeline, args)
        writer.add_scalar('accuracy', score, global_step=(i + 1) * iters)
        print(f'accuracy at epoch: {i + 1} is {score}')
        if score > best:
            best = score
            if args.save_model:
                print(f'save best model at epoch {i}')
                if not osp.exists(args.ckpt_dir):
                    os.makedirs(args.ckpt_dir)
                torch.save(model.state_dict(), osp.join(args.ckpt_dir, args.ckpt_name))


def val(model, data_pipeline, args):
    dataset = ItemDataset(args.val_file, data_pipeline, with_label=True, allowed_keys=args.allowed_keys)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=True, drop_last=False)

    model.eval()

    results = []
    labels = []
    val_pb = ProgressBar(len(dataloader))
    val_pb.start()
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            data, label = batch

            label = label.detach().numpy()
            for key in data['text']:
                data['text'][key] = data['text'][key].squeeze().cuda()
            output = model(data).detach().cpu().numpy().argmax(1)

            results.append(output)
            labels.append(label)
            val_pb.update()
    labels = np.concatenate(labels, axis=0)
    results = np.concatenate(results, axis=0)
    score = f1_score(labels, results, average='macro')
    return score


def test(args):
    data_pipeline = DataProcessPipeline(args.bert_path, args.allowed_keys)
    dataset = ItemDataset(args.test_file, data_pipeline, with_label=False, allowed_keys=args.allowed_keys)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=True, drop_last=False)

    bert = BERT(pretrained=args.bert_path, freeze=True)
    mlp = MLP(layer_num=args.mlp_layer_num, dims=args.mlp_dims, with_bn=args.with_bn, act_type=args.act_type,
              last_w_bnact=args.last_w_bnact, last_w_softmax=args.last_w_softmax)
    model = Classifier(bert, mlp).cuda()
    model.load_state_dict(torch.load(osp.join(args.ckpt_dir, args.ckpt_name)))
    model.eval()

    results = []
    test_pb = ProgressBar(len(dataloader))
    test_pb.start()
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            data = batch

            for key in data['text']:
                data['text'][key] = data['text'][key].cuda()
            output = model(data).detach().cpu().numpy().argmax(1)
            results.append(output)
            test_pb.update()
    results = np.concatenate(results, axis=0)
    if not osp.exists(args.result_dir):
        os.makedirs(args.result_dir)
    with open(osp.join(args.result_dir, args.result_name), 'w', encoding='utf-8') as f:
        f.write('\n'.join([dataset.idx2label[result] for result in results]))


def main():
    args = parse()
    if args.mode == 'train':
        train(args)
    else:
        test(args)


if __name__ == '__main__':
    main()
