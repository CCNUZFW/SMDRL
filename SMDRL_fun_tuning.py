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
from torchinfo import summary
device = try_gpu()
torch.manual_seed(42)
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

state_dict = torch.load('./CCNU516_base40_TP.pt')##加载保存的模型
newmodel = Encoder().to(device=device)
newmodel.load_state_dict(state_dict)

classfier = classfier()
# classfier = ECAPA_TDNN()
classfier.to(device=device)
train_file="./CCNU_train_150_256_label.csv"###训练数据集
train_dataset=ClassfierDataset(train_file)

train_size=int(len(train_dataset)*0.8)
val_size = len(train_dataset)-train_size
train_data,val_data = random_split(train_dataset,[train_size,val_size])
train_lodar=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=0)
val_lodar = DataLoader(val_data,shuffle=True)


test_file="./CCNU_test_100_256_label.csv"##测试数据集
test_dataset=ClassfierDataset(test_file)
test_lodar=DataLoader(test_dataset,batch_size=100,shuffle=True,num_workers=0)

newmodel.eval()

optimizer = torch.optim.Adam(classfier.parameters(),lr=1e-4,weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

total_train_step = 0
total_test_step = 0
torch.backends.cudnn.enabled = False
start_time = time.time()

epoch_list=[]
train_loss_list=[]
test_loss_list=[]
test_acc=[]

for epoch in range(200):
    print("---------------第{}轮训练开始-------------------".format(epoch+1))
    classfier.train()
    ##训练步骤开始
    total_train_loss=0
    for step,(x,y) in enumerate(tqdm(train_lodar)):
         inputs , label = (x,y)
         inputs = torch.reshape(inputs, (-1, 1, 256, 39))
         inputs = inputs.to(device=device)
         label = label.to(device=device)
         label = label.squeeze(-1)

         online_encoderoutput = newmodel(inputs)
         online_encoderoutput = online_encoderoutput.unsqueeze(dim=1)
         out = classfier(online_encoderoutput)

         train_loss= criterion(out,label.long())
         total_train_loss+=train_loss

         optimizer.zero_grad()
         train_loss.backward()
         optimizer.step()

         total_train_step+=1
    print("Loss:{}".format(total_train_loss.item()))

    ##验证步骤开始
    classfier.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (x, y) in enumerate(val_lodar):
            inputs, label = (x, y)
            inputs = torch.reshape(inputs, (-1, 1, 256, 39))
            inputs = inputs.to(device=device)
            label = label.to(device=device)
            label = label.squeeze(-1)

            online_encoderoutput = newmodel(inputs)
            online_encoderoutput = online_encoderoutput.unsqueeze(dim=1)
            out = classfier(online_encoderoutput)

            loss = criterion(out, label.long())
            total_val_loss+=loss
            res = torch.softmax(out.data, dim=1)
            _, pred_y = torch.max(res, dim=1)
            total += y.size(0)
            correct += (pred_y == label).sum().item()
    print("验证集上的Loss:{}".format(total_val_loss))
    print("验证集上的正确率:{}".format(correct/total))


    epoch_list.append(epoch+1)
    train_loss_list.append(total_train_loss.item())
    test_loss_list.append(total_val_loss.item())
    test_acc.append(correct/total)

torch.save(classfier.state_dict(),'设置你想要保存的名字.pt')
print("模型已保存")

epoch_list = pd.DataFrame(epoch_list)
train_loss_list = pd.DataFrame(train_loss_list)
test_loss_list = pd.DataFrame(test_loss_list)
test_acc = pd.DataFrame(test_acc)

res =pd.DataFrame(pd.concat([epoch_list,train_loss_list,test_loss_list,test_acc],axis=1))
res.to_csv('设置你想要保存的名字.csv',index=False,header=['epoch','train_loss','val_loss','val_acc'],float_format='%.6f')
print("训练数据已保存")
