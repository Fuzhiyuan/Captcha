import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel=32
        self.conv=nn.Conv2d(1,self.channel,3,3,padding=1)
        self.conv2=nn.Conv2d(self.channel,self.channel*2,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(self.channel*2,self.channel*3,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(self.channel*3,self.channel*4,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.LeakyReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.relu(x)
        x= self.conv4(x)
        x = self.pool(x)
        x = self.relu(x)

        shape_size = x.shape[1]*x.shape[2]*x.shape[3]
        batch_size=x.shape[0]
        x = x.view(x.shape[0], -1)
        
        fc = nn.Linear(shape_size,64*1*5)
        output = fc(x)
        output = output.view(batch_size,64,1,5)
        return output


