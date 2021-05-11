import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoModel


class BERT(nn.Module):
    def __init__(self, pretrained=None, freeze=True):
        super(BERT, self).__init__()
        self.pretrained = pretrained
        self.freeze = freeze

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            self.model = AutoModel.from_pretrained(self.pretrained).to('cuda')
            # 是否要固定住模型
            # self.model.eval()
            # # self.model.train()
        else:
            raise TypeError('pretrained must be a str')

    def forward(self, x):

        if self.freeze:
            with torch.no_grad():
                text_out = self.model(**x).pooler_output
        else:
            text_out = self.model(**x).pooler_output
        return text_out


class MLP(nn.Module):

    def __init__(self, layer_num, dims=(1, 1), with_bn=True, act_type='relu', last_w_bnact=False):
        super().__init__()
        assert layer_num == len(dims) - 1, "unmatched layer parameters!"
        layers = []
        for i in range(layer_num - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if with_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if act_type == 'relu':
                layers.append(nn.ReLU())
            else:
                raise ValueError(f'There is no {act_type} implementation now!')
        layers.append(nn.Linear(dims[layer_num - 1], dims[layer_num]))
        if last_w_bnact:
            if with_bn:
                layers.append(nn.BatchNorm1d(dims[layer_num]))
            if act_type == 'relu':
                layers.append(nn.ReLU())
            else:
                raise ValueError(f'There is no {act_type} implementation now!')
        self.model = nn.Sequential(*layers)

    def init_weight(self):
        pass

    def forward(self, x):
        out = self.model(x)
        return out


class Classifier(nn.Module):
    # bert, mlp, head
    def __init__(self, bert, mlp):
        super().__init__()
        self.bert = bert
        self.mlp = mlp
        self.loss = nn.CrossEntropyLoss()

    def forward_train(self, x, label):
        loss = self.loss(x, label)
        return loss

    def forward_test(self, x):
        pass

    def forward(self, x, label=None):
        if label:
            return self.forward_train(x, label)
        else:
            return self.forward_test(x)


# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# for input, target in dataset:
#     optimizer.zero_grad()
#     output = model(input)
#     loss = loss_fn(output, target)
#     loss.backward()
#     optimizer.step()
