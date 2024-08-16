import  torch
import csv
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset,random_split
from torch import nn
import torch.backends
from sklearn.metrics import confusion_matrix, classification_report
import time
from SMDRL_classfier import classfier
from CMDEncoder import Encoder,try_gpu
from tqdm import tqdm
from torch.nn import functional as F
from ecapa_tdnn import ECAPA_TDNN
import  time
from sklearn.metrics import roc_curve,auc,precision_recall_fscore_support
import matplotlib.pyplot as plt
torch.manual_seed(42)
device = try_gpu()
class ClassfierDataset(Dataset):
    def __init__(self,filepath):
        data=np.loadtxt(filepath,delimiter=',',dtype=np.float32,encoding='UTF-8')
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:,:-1])##训练数据
        self.y_data = torch.from_numpy(data[:,[-1]])##label
        print('数据准备正常')
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

state_dict = torch.load('./CCNUPLUS516_base40_TP.pt')##加载基础模型
newmodel = Encoder().to(device=device)
newmodel.load_state_dict(state_dict)
classfier = classfier()
#classfier = ECAPA_TDNN()
cla_state_dict = torch.load("./CCNUPLUS516_base40_funtuning_TP_classfier_200ep.pt")##加载微调模型
classfier.load_state_dict(cla_state_dict)
classfier.to(device=device)

test_file="./CCNUPLUS_test_100_256_label.csv"##测试集
test_dataset=ClassfierDataset(test_file)
# print(dataset[0])
test_lodar=DataLoader(test_dataset,batch_size=128,shuffle=True,num_workers=0)
newmodel.eval()
correct = 0
total = 0

with torch.no_grad():
    for step, (x, y) in enumerate(test_lodar):
        inputs, label = (x, y)
        inputs = torch.reshape(inputs, (-1, 1, 256, 39))
        inputs = inputs.to(device=device)
        label = label.to(device=device)
        label = label.squeeze(-1)

        online_encoderoutput = newmodel(inputs)
        online_encoderoutput = online_encoderoutput.unsqueeze(dim=1)
        out = classfier(online_encoderoutput)
        res = torch.softmax(out.data, dim=1)
        _, pred_y = torch.max(res, dim=1)
        total += y.size(0)
        correct += (pred_y == label).sum().item()
    print("测试集上的正确率:{:.6f}".format(correct / total))
