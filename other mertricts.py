import  torch
import csv
import numpy as np
import pandas as pd
import torchaudio.transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
import queue
import torch.backends
from sklearn.metrics import confusion_matrix, classification_report
import time
from torch.utils.tensorboard.writer import SummaryWriter
import pandas as pd
from CMDEncoder import Encoder,try_gpu
from SMDRL_classfier import classfier
import time
import matplotlib.pyplot as plt
from ecapa_tdnn import ECAPA_TDNN
torch.manual_seed(42)
device = try_gpu()
state_dict = torch.load('./CCNUPLUS516_base40_TP.pt')##加载基础模型
newmodel = Encoder().to(device=device)
newmodel.load_state_dict(state_dict)
classfier = classfier()
# classfier = ECAPA_TDNN()
cla_state_dict = torch.load("./CCNUPLUS516_base40CNN2_funtuning_TP_classfier_200ep.pt")##加载微调模型
classfier.load_state_dict(cla_state_dict)
classfier.to(device=device)

y_test = np.zeros((60*100, 60), 'float')
for i in range(60):
    y_test[i * 100:(i + 1) * 100, i] = 1

# y_test = np.zeros((55 * 100, 55), 'float')
# for i in range(55):
#     y_test[i * 100:(i + 1) * 100, i] = 1
# y_test = np.zeros((45 * 100, 45), 'float')
# for i in range(45):
#     y_test[i * 100:(i + 1) * 100, i] = 1

path = './CCNUPLUS_test_100_256_label.csv'##测试集
inputs_test = pd.read_csv(path, header=None)
test_input = inputs_test.values
test_input = test_input[:,:-1]
x_test = test_input.reshape((test_input.shape[0], 1,256, 39))
x_test = torch.FloatTensor(x_test).to(device)

start = time.time()
o1 = newmodel(x_test)
o1 = o1.unsqueeze(dim=1)

o2 = classfier(o1)
end = time.time()
for i in range(len(o2)):
    max_value = max(o2[i])
    for j in range(len(o2[i])):
        if max_value == o2[i][j]:
            o2[i][j] = 1
        else:
            o2[i][j] = 0
# y_test = y_test.cpu().numpy()
o2 = o2.detach().cpu().numpy()
print('classification report', classification_report(y_test, o2, digits=6))
print(end-start)

