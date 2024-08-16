import torch
import csv
import numpy as np
import pandas as pd
import torchaudio.transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
import queue


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.device = try_gpu()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=3),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )
        self.BiLSTM = nn.LSTM(input_size=16128, hidden_size=128, batch_first=True, bidirectional=True)
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = Reshape(x.shape[0], 1, x.shape[1] * x.shape[2] * x.shape[3])(x)


        # x = Reshape(x.shape[0],  x.shape[1] * x.shape[2] * x.shape[3])(x)
        # ##消融CNN加的MLP
        # x = self.linear1(x)
        # x = self.linear2(x)


        out, (h_n, c_n) = self.BiLSTM(x)
        x = torch.permute(h_n, (1, 0, 2)).contiguous()
        x = Reshape(x.shape[0], x.shape[1] * x.shape[2])(x)

        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        mean = torch.mean(x, dim=0)
        std = torch.sqrt(x.clamp(min=1e-9))
        x2 = x + mean + std
        return x2


