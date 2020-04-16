import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import data_preprocess
import VGG16

x_train,y_train,x_valid,y_valid = data_preprocess.load_data()

VGG_16 = VGG16.get_model()

def get_model():
    pass

lr = 0.001
epochs = 2
loss_function = 