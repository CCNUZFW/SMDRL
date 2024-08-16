import torch
import  librosa
from librosa import feature
import numpy as np
import matplotlib.pyplot as plt
import torchaudio.transforms as T
from torch import nn
import random
from  scipy.io import wavfile

#######自动搜素device
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
device = try_gpu()
####计算插值系数
def Interpolation_factor(size,concentration_0=0.2, concentration_1=0.3):

    gamma_1_sample = torch.distributions.gamma.Gamma(concentration=2.0,rate=concentration_0)
    a=gamma_1_sample.sample(sample_shape=(size[0],size[1],size[2],size[3]))##从gamma分布中提取数1
    a=a.numpy()

    gamma_2_sample = torch.distributions.gamma.Gamma(concentration=2.0, rate=concentration_1)
    b=gamma_2_sample.sample(sample_shape=(size[0],size[1],size[2],size[3]))##从gamma分布中提取数2
    b=b.numpy()
    return torch.tensor(a/ (a + b))##结合两个数，计算出最终插值系数
#############混入高斯噪声
def MixGaussianNoise(data ,ratio):
    x=torch.exp(data).cpu()
    lambd = ratio*np.random.rand()##[0,1)均匀分布随选
    z = torch.exp(torch.normal(0,lambd,size=x.shape))
    mixed = (1 - lambd) * x + z + torch.finfo(data.dtype).eps##一个很小的正数，防止分母为0
    return torch.log(mixed)

############增强1:插值混噪#########
def Interpolative_noise_mixing(sample,alpha=0.2):
    x1=sample.cpu()
    x2=MixGaussianNoise(x1,alpha)###先混高斯噪声
    gamma=Interpolation_factor((sample.shape[0],sample.shape[1],sample.shape[2],sample.shape[3]))
    x3 = x1 * gamma + x2 * (1-gamma)##再插值
    return torch.tensor(x3).to(device)##x3就是增强后的数据

############增强2：时频双掩码#########
def T_F_mask(tensor,mask_len):##时频双掩码
    masking1 = T.TimeMasking(time_mask_param=mask_len).to(device)
    masking2 = T.FrequencyMasking(freq_mask_param=mask_len).to(device)
    masked = masking1(tensor)
    masked = masking2(masked)
    return masked.to(device)

###########增强3：分割重采样###############
'''
    将一段音频按比例分成两段，再按照一定的重采样率各自采样，再拼接
'''
def Partitioned_resampling(tensor):
    audio1 = tensor[:, :, :, :26]  ###3:2近似黄金分割比
    audio2 = tensor[:, :, :, 26:]
    trans1 = T.Resample(16000, 12800).to(device)  ##3a+2b = 5
    trans2 = T.Resample(16000, 20800).to(device)
    a1 = trans1(audio1)
    a2 = trans2(audio2)
    return torch.cat([a1, a2[:, :, :, :]], dim=3).to(device)


