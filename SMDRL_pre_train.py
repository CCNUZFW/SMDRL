import  torch
import csv
import numpy as np
import pandas as pd
import torchaudio.transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

import queue
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from Augmentation import try_gpu,Interpolative_noise_mixing,T_F_mask,Partitioned_resampling
from CMDEncoder import Encoder

torch.manual_seed(42)


###L2正则化损失
class Loss_Fn(nn.Module):##自定义L2-normalized损失函数
    def __init__(self):
        super(Loss_Fn, self).__init__()
    def forward(self,x,y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return torch.mean(2 - 2 * (x * y).sum(dim=-1))##按行取了平均值

####模型搭建###
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)

class Project(nn.Module):
    def __init__(self):
        super(Project,self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,60)   ######最终分类数，要结合数据集设置
        )
    def forward(self,x):
        x=self.layers(x)
        return x
class Prediction(nn.Module):
    def __init__(self):
        super(Prediction, self).__init__()
    def forward(self, x):
        x = torch.softmax(x, dim=1)
        return x

# class Prediction(nn.Module):
#     def __init__(self):
#         super(Prediction, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(60, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 60)
#         )
#     def forward(self,x):
#         x= self.net(x)
#         #x = torch.softmax(x,dim=1)
#         return x

class My_Byol(nn.Module):
    def __init__(self):
        super(My_Byol, self).__init__()
        self.onlin_encoding = Encoder()
        self.onlin_projection = Project()
        self.onlin_prediction = Prediction()

        self.target_encoding = Encoder()
        self.target_projection = Project()

    def forward(self,x):
        ##########在此更改两种不同的数据增强方式###############
        online_input = T_F_mask(x,50)
        target_input = Partitioned_resampling(x)


        ##online网络
        o1 = self.onlin_encoding(online_input)
        o2 = self.onlin_projection(o1)
        o3 = self.onlin_prediction(o2)

        ##target网络
        t1 = self.target_encoding(target_input)
        t2 = self.target_projection(t1)
        return o1,o3,t2
class MyDataset(Dataset):
    def __init__(self,filepath):
        data=np.loadtxt(filepath,delimiter=',',dtype=np.float32,encoding='UTF-8')
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:,:])##训练数据
        print('数据准备正常')
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        # return self.x_data[index],self.y_data[index]
            return self.x_data[index]

##数据准备
device=try_gpu()
model = My_Byol().to(device=device)
summary(model,input_size=(5,1,256,39))
file="./CCNUPLUS_base_516_256.csv"##基础模型训练数据集
train_dataset= MyDataset(file)
train_lodar=DataLoader(train_dataset,batch_size=180,shuffle=True,num_workers=0)
criterion = Loss_Fn()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
epoch_list=[]
train_loss_list=[]

#############训练流程###########
for epoch in range(40):
    model.train()
    total_train_loss = 0
    for step,x in enumerate(tqdm(train_lodar)):
        torch.cuda.empty_cache()
        inputs = x
        inputs = torch.reshape(inputs, (-1, 1, 256, 39)).to(device)

        ##计算损失
        online_encoderoutput , online_preoutput , target_out = model(inputs)
        loss = criterion(online_preoutput,target_out)

        ##对称损失
        # _online_encoderoutput, _online_preoutput, _target_out = model(x2, x1)
        # loss2 = criterion(_online_preoutput, _target_out)
        # loss = loss1+loss2

        total_train_loss+=loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: ",epoch+1,"Loss: ","{:.10f}".format(total_train_loss))
    epoch_list.append(epoch + 1)
    train_loss_list.append(total_train_loss.item())

#######保存训练数据######
torch.save(model.onlin_encoding.state_dict(),'设置想要保存的名字.pt')
print("模型保存成功")
epoch_list = pd.DataFrame(epoch_list)
train_loss_list = pd.DataFrame(train_loss_list)

res =pd.DataFrame(pd.concat([epoch_list,train_loss_list],axis=1))
res.to_csv('设置想要保存的名字.csv',index=False,header=['epoch','train_loss'],float_format='%.6f')
print("训练数据保存成功")