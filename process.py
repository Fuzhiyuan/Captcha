import cv2 as cv
import numpy as np
import os

def function(file):
    image = cv.imread(file,0)
    binary_image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    binary_image=255-binary_image
    num,label,inf,inf_c = cv.connectedComponentsWithStats(binary_image,connectivity=8)
    count_connect = inf[:,-1]
    for i in range(1,num):
        if(count_connect[i]<1000):
            binary_image[label==i]=0

    return binary_image
if __name__=="__main__":
    path ="D:/CodeRepo/sample-code/data/train/"
    new_path = "D:/CodeRepo/sample-code/data/Newtrain/"
    if os.path.isdir(new_path)!=True:
        os.makedirs(new_path)

    file_list = os.listdir(path)
    for f in file_list:
        print("contemporary file is ",f)
        img = function(path+f)
        cv.imwrite(new_path+f,img)
