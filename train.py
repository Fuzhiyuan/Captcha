import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from Network import Net
from torch.utils.data import DataLoader
from DataLoader import MyDataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import time



# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    root_path = ""
    trainDataset_path = root_path+'/train/'
    save_model_path=root_path+'/model/'
    Load_flag=False
    load_model_path=root_path+'/model/'
    log_dir = root_path+"/log/"+str(time.time())
    writer = SummaryWriter(log_dir)
    learning_rate=0.0001
    betas = (0.9,0.999)
    eps=1e-8
    weight_decay = 1e-4
    load_epoch=0
    Network = Net()
    optimizer = optim.Adam(
        Network.parameters(),
        lr=0.001,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )
    criteria=nn.CrossEntropyLoss()
    if Load_flag:
        model_dict = torch.load(load_model_path)
        Network.load_state_dict(model_dict['model'])
        optimizer.load_state_dict(model_dict['optimizer'])
        load_epoch = model_dict['epoch']

    trainDataset = MyDataset(trainDataset_path)
    batch_size=32
    trainDataLoader=DataLoader(trainDataset,batch_size,shuffle=True)
    Epoch=100
    count=0
    for i in range(load_epoch,Epoch):
        for batch in trainDataLoader:
            optimizer.zero_grad()
            train = batch['input']
            target = batch['target']
            output = Network(train)
            loss =criteria(output,target)
            loss.backward()
            writer.add_scalar('Loss',loss.item(),global_step=i*len(trainDataLoader)+count)
            #print("Epoch is ",i,"batch is ","now the loss is ",loss,)
            optimizer.step()
            count+=1
        torch.save(
            {
                'model':Network.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':i
            },save_model_path+str(i)+"-model.pt"
        )