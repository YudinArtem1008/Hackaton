#from torchvision import models
#from numpy import genfromtxt
#import torch
#import cv2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#print(model)
#my_data = genfromtxt('my_file.csv', delimiter=',')
#df_ndvi = pd.read_csv("train/NDVI.csv", sep=";", encoding="windows-1251")
#print(df_ndvi)

#model = RandomForestClassifier(n_estimators=10,
#                               oob_score=True,
#                               random_state=1)
#data = np.array(pd.read_csv("aboba.csv", sep=','))[0:4, :]
#print(data)


print(np.array(x).argmax(0))