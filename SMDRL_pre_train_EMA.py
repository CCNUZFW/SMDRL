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
from Augmentation import try_gpu,Interpolative_noise_mixing,T_F_mask,Partitioned_resampling
from CMDEncoder import Encoder
from tqdm import tqdm
import torch
from torch.nn.functional import cosine_similarity
class EMA:
    def __init__(self, model, target_model, decay=0.99):
        self.model = model
        self.target_model = target_model
        self.decay = decay
        self.shadow_params = {}
        self.init_shadow_params()

    def init_shadow_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()

    def update_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = self.decay * self.shadow_params[name] + (1 - self.decay) * param.data

        for name, param in self.target_model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow_params[name]


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
            nn.Dropout(0.4),
            nn.Linear(256,60)   ######最终分类数，要结合数据集设置
        )
    def forward(self,x):
        x=self.layers(x)
        return x

class Prediction(nn.Module):
    def __init__(self):
        super(Prediction, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(60, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 60)
        )
    def forward(self,x):
        x= self.net(x)
        return x

class Online_model(nn.Module):
    def __init__(self):
        super(Online_model,self).__init__()
        self.encoding = Encoder()
        self.projection = Project()
        self.prediction = Prediction()
    def forward(self,x):
        ##online网络
        o1 = self.encoding(x)
        o2 = self.projection(o1)
        o3 = self.prediction(o2)
        return o1,o3

class Target_model(nn.Module):
    def __init__(self):
        super(Target_model,self).__init__()
        self.encoding = Encoder()
        self.projection = Project()
    def forward(self,x):
        t1 = self.encoding(x)
        t2 = self.projection(t1)
        return t2

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
online_model = Online_model().to(device=device)
target_model = Target_model().to(device=device)

file="./CCNUPLUS_base_516_256.csv"##基础模型训练数据集
train_dataset= MyDataset(file)

train_lodar=DataLoader(train_dataset,batch_size=180,shuffle=True,num_workers=0)
criterion = Loss_Fn()

online_optimizer = torch.optim.Adam(online_model.parameters(),lr=1e-4,weight_decay=1e-4)
target_optimizer = EMA(online_model,target_model,decay=0.99)
epoch_list=[]
train_loss_list=[]

#############训练流程###########
for epoch in range(40):
    online_model.train()
    total_train_loss = 0
    for step,x in enumerate(tqdm(train_lodar)):
        torch.cuda.empty_cache()
        inputs = x
        inputs = torch.reshape(inputs, (-1, 1, 256, 39)).to(device=device)

        ####两种数据增强方式######
        x1 = T_F_mask(inputs, 50)
        x2 = Partitioned_resampling(inputs)

        online_encoderoutput , online_preoutput  = online_model(x1)
        target_out = target_model(x2)
        loss1 = criterion(online_preoutput,target_out)

        _online_encoderoutput, _online_preoutput = online_model(x2)
        _target_out = target_model(x1)
        loss2 = criterion(_online_preoutput, _target_out)
        loss = loss1+loss2
        total_train_loss+=loss

        online_optimizer.zero_grad()
        loss.backward()

        online_optimizer.step()
        target_optimizer.update_params()

    print("Epoch: ",epoch+1,"Loss: ","{:.10f}".format(total_train_loss))
    epoch_list.append(epoch + 1)
    train_loss_list.append(total_train_loss.item())

#######保存训练数据######
torch.save(online_model.encoding.state_dict(),'设置想要保存的名字.pt')
print("模型保存成功")
epoch_list = pd.DataFrame(epoch_list)
train_loss_list = pd.DataFrame(train_loss_list)

res =pd.DataFrame(pd.concat([epoch_list,train_loss_list],axis=1))
res.to_csv('设置想要保存的名字.csv',index=False,header=['epoch','train_loss'])
print("训练数据保存成功")