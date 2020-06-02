import torch.nn as nn
import backbone
from utils import Linear as Linear
from torch.nn import functional as F
import torch

class Network(nn.Module):
    def __init__(self, num_classes, projection_size, base_model, pretrained, normalize, contrastive, linear_size):
        super(Network, self).__init__()
        self.train_final = False
        self.contrastive = contrastive
        self.e = Encoder(base_model, pretrained, normalize)
        feature_size = self.e.removed

        if contrastive:
            self.p = Projection(projection_size, normalize, linear_size, feature_size)
        self.o = Output(num_classes, feature_size)
        self.frozen = False
        self.is_train = True

    def forward(self, x):
        z = self.e(x)
        if not self.train_final and self.contrastive:
            f = self.p(z)
        else:
            f = self.o(z)
        return f

    def convert(self):
        self.train_final = ~self.train_final

    def freeze(self):
        freeze(self.e, not self.frozen)

    def eval(self):
        self.is_train = False

    def train(self):
        self.is_train = True



class Encoder(nn.Module):
    def __init__(self, base_model, pretrained, normalize):
        super(Encoder, self).__init__()
        self.normalize = normalize
        self.backbone = backbone.ResnetBackbone(base_model, pretrained)
        self.removed = self.backbone.fc.weight.shape[1]
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))

    def forward(self, x):
        x = self.backbone(x)
        if self.normalize:
            x = F.normalize(x)
        return x


class Projection(nn.Module):
    def __init__(self, output_num, normalize, linear_size, input_size):
        super(Projection, self).__init__()
        self.normalize = normalize
        self.l1 = Linear(input_size, linear_size, bn=False, activ='relu')
        self.l2 = Linear(linear_size, output_num, bn=False, activ=None)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        x = self.l2(x)
        if self.normalize:
            x = F.normalize(x)
        return x


class Output(nn.Module):
    def __init__(self, num_classes, input_size):
        super(Output, self).__init__()
        self.l1 = Linear(input_size, num_classes, bn=False, activ=None)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        return x


def freeze(module, value=False):
    for param in module.parameters():
        param.requires_grad = value

