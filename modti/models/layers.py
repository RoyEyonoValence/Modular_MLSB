import torch
import torch.nn as nn
from modti.utils import get_activation


class Cosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2)


class SquaredCosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2) ** 2


class DotProduct(nn.Module):
    def forward(self, x1, x2):
        return (x1 * x2).sum(-1)


class Euclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0)


class SquaredEuclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0) ** 2


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:

    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer. Should be one supported by :func:`ivbase.nn.commons.get_activation`.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{node_feats_dim}}`
            (Default value = None)

    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer

    """

    def __init__(self, in_size, out_size, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None):
        super(FCLayer, self).__init__()
        # Although I disagree with this it is simple enough and robust
        # if we trust the user base
        self._params = locals()
        self.in_size = in_size
        self.out_size = out_size
        activation = get_activation(activation)
        linear = nn.Linear(in_size, out_size, bias=bias)
        if init_fn:
            init_fn(linear)
        layers = [linear, activation]
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        if b_norm:
            layers.append(nn.BatchNorm1d(out_size))
        self.net = nn.Sequential(*layers)

    @property
    def output_dim(self):
        return self.out_size

    @property
    def out_features(self):
        return self.out_size

    @property
    def in_features(self):
        return self.in_size

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    r"""
    Feature extractor using a Fully Connected Neural Network

    Arguments
    ----------
        input_size: int
            size of the input
        hidden_sizes: int list or int
            size of the hidden layers
        activation: str or callable
            activation function. Should be supported by :func:`ivbase.nn.commons.get_activation`
            (Default value = 'relu')
        b_norm: bool, optional):
            Whether batch norm is used or not.
            (Default value = False)
        dropout: float, optional
            Dropout probability to regularize the network. No dropout by default.
            (Default value = .0)

    Attributes
    ----------
        extractor: torch.nn.Module
            The underlying feature extractor of the model.
    """

    def __init__(self, input_size, hidden_sizes, activation='ReLU', b_norm=False, l_norm=False, dropout=0.0):
        super(MLP, self).__init__()
        self._params = locals()
        layers = []
        in_ = input_size
        if l_norm:
            layers.append(nn.LayerNorm(input_size))
        for i, out_ in enumerate(hidden_sizes):
            layer = FCLayer(in_, out_, activation=activation,
                            b_norm=b_norm and (i == (len(hidden_sizes) - 1)),
                            dropout=dropout)
            layers.append(layer)
            in_ = out_

        self.__output_dim = in_
        self.extractor = nn.Sequential(*layers)

    @property
    def output_dim(self):
        return self.__output_dim

    def forward(self, x):
        res = self.extractor(x)
        return res


class DeepConcat(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_sizes, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        self.fc = MLP(x1_dim+x2_dim, hidden_sizes, activation=activation, dropout=dropout)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        return self.fc(x).squeeze()


AVAILABLE_LAYERS = {
    "Cosine": Cosine,
    "SquaredCosine": SquaredCosine,
    "Euclidean": Euclidean,
    "SquaredEuclidean": SquaredEuclidean,
    "DotProduct": DotProduct,
    "DeepConcat": DeepConcat,
}
