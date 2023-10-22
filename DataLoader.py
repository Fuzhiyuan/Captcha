import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import cv2 as cv
import os
import torch


class MyDataset(Dataset):
    def __init__(self,train_dir):
        super().__init__()
        self.index="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.root=train_dir
        self.list = os.listdir(train_dir)

    def __len__(self):
        return len(self.list)
    def __getitem__(self, item):
        file = self.list[item]
        label_string = file.split(".")[0]
        train_img = cv.imread(self.root+file,0)
        train = torch.from_numpy(train_img)
        label_list = []
        for i in label_string:
            idx = self.index.find(i)
            label_list.append(idx)
        label = torch.tensor(label_list,dtype=torch.int64)


    # def GetList(self,train_dir):
    #     train_list = os.listdir(train_dir)
    #     #varify_list = os.listdir(varify_dir)
    #     train_dict = {}
    #     varify_dict={}
    #     for L in train_list:
    #         img = cv.imread(L,0)
    #         img_tensor = torch.from_numpy(img)
    #         img_tensor=img_tensor.float()/255.
    #         train_dict[L]=img_tensor
    #     print("train data load successfully")



