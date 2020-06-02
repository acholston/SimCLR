import torch.nn as nn

def normalize(var):
    norm = var.norm(p=2, dim=1, keepdim=True)
    return var.div(norm)


def activations(activ):
    activations = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1)}
    return activations[activ]


class Linear(nn.Module):
    def __init__(self, num_in, num_out, bn=False, activ=None):
        super(Linear, self).__init__()
        self.bn = bn
        self.activ = activ

        self.layer = nn.Linear(num_in, num_out)
        nn.init.normal_(self.layer.weight, 0, 0.1)
        nn.init.constant_(self.layer.bias, 0)
        if bn:
            self.bn = nn.BatchNorm1d(num_out)
            nn.init.normal_(self.bn.weight, 0, 1)
            nn.init.constant_(self.bn.bias, 0)
        if activ is not None:
            self.activ = activations(activ)

    def forward(self, x):
        x = self.layer(x)
        if self.bn:
            x = self.bn(x)
        if self.activ is not None:
            return self.activ(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=3, stride=1, padding=0, dilation=1, bn=False, activ=None, bias=True):
        super(Conv2d, self).__init__()
        self.bn = bn
        self.activ = activ

        self.layer = nn.Conv2d(num_in, num_out, kernel_size=kernel_size, stride=stride , padding=padding, dilation=dilation, bias=bias)
        nn.init.normal_(self.layer.weight, 0, 0.1)
        if bias:
            nn.init.constant_(self.layer.bias, 0)
        if bn:
            self.bn = nn.BatchNorm2d(num_out)
            nn.init.normal_(self.bn.weight, 0, 1)
            nn.init.constant_(self.bn.bias, 0)
        if activ is not None:
            self.activ = activations(activ)

    def forward(self, x):
        x = self.layer(x)
        if self.bn:
            x = self.bn(x)
        if self.activ is not None:
            return self.activ(x)
        return x
