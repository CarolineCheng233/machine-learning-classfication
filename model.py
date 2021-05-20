import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoModel


class BERT(nn.Module):
    def __init__(self, pretrained=None, freeze=True):
        super(BERT, self).__init__()
        self.pretrained = pretrained
        self.freeze = freeze

    def init_weight(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            self.model = AutoModel.from_pretrained(self.pretrained)
            # 是否要固定住模型
        else:
            raise TypeError('pretrained must be a str')

    def forward(self, x):
        if self.freeze:
            self.model.eval()
            with torch.no_grad():
                text_out = self.model(**x).pooler_output
        else:
            text_out = self.model(**x).pooler_output
        return text_out


class MLP(nn.Module):

    def __init__(self, layer_num, dims=(1, 1), with_bn=True, act_type='relu', last_w_bnact=False, last_w_softmax=True):
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
        self.last_w_softmax = last_w_softmax
        if last_w_softmax:
            self.softmax = nn.Softmax(dim=1)
        self.model = nn.Sequential(*layers)

    def init_weight(self):
        nn.init.normal_(self.model.weight, std=0.02)

    def forward(self, x):
        out = self.model(x)
        if self.last_w_softmax:
            out = self.softmax(out)
        return out


class Classifier(nn.Module):
    # bert, mlp
    def __init__(self, bert, mlp):
        super().__init__()
        self.bert = bert
        self.mlp = mlp
        self.init_weight()

    def init_weight(self):
        self.bert.init_weight()
        self.mlp.init_weight()

    def forward(self, summary):
        # for key in summary:
        #     summary[key] = summary[key].reshape((-1,) + summary[key].shape[2:])
        output = self.bert(summary)
        output = self.mlp(output)
        return output


# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# for input, target in dataset:
#     optimizer.zero_grad()
#     output = model(input)
#     loss = loss_fn(output, target)
#     loss.backward()
#     optimizer.step()
