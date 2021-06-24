"""The lightweight LSTM network model for PyTorch used in FedDA.

Reference:

Zhang et al., "Dual Attention-Based Federated Learning for Wireless Traffic Prediction,"
in 2021 IEEE Conference on Computer Communications (INFOCOM).

https://chuanting.github.io/assets/pdf/ieee_infocom_2021.pdf

https://github.com/chuanting/FedDA
"""
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class WeightLayer(nn.Module):
    def __init__(self):
        super(WeightLayer, self).__init__()
        self.w = nn.Parameter(torch.rand(1, requires_grad=True))

    def forward(self, x):
        return x * self.w


class Model(nn.Module):
    """The lightweight LSTM network model."""
    def __init__(self):
        super().__init__()
        # input feature dimension of LSTM
        self.input_dim = 1
        # hidden neurons of LSTM layer
        self.hidden_dim = 64
        # number of layers of LSTM
        self.num_layers = 2
        self.out_dim = 1

        self.lstm_close = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, dropout=0.2)
        self.lstm_period = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                   batch_first=True, dropout=0.2)

        self.weight_close = WeightLayer()
        self.weight_period = WeightLayer()

        self.linear_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, xc, xp=None):
        """Forward pass."""
        bz = xc.size(0)
        h0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim))
        c0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim))

        self.lstm_close.flatten_parameters()
        self.lstm_period.flatten_parameters()

        xc_out, xc_hn = self.lstm_close(xc, (h0, c0))
        x = xc_out[:, -1, :]
        xp_out, xp_hn = self.lstm_period(xp, (h0, c0))
        y = xp_out[:, -1, :]
        out = x + y
        y_pred = self.linear_layer(out)
        return y_pred

    @staticmethod
    def get_model(*args):
        """Obtaining an instance of this model."""
        return Model()




        



