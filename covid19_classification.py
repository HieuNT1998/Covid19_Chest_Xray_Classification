import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import data_preprocess
import VGG16
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_train,y_train,x_valid,y_valid = data_preprocess.load_data()
VGG_16_model = VGG16.get_model()

lr = 0.001
epochs = 2
loss_function = F.cross_entropy
bs = 16
optim = torch.optim.SGD(VGG_16_model.parameters(),lr = lr)

train_ds = list(zip(x_train,y_train))
train_dl = DataLoader(train_ds,batch_size=16)

valid_ds = list(zip(x_train,y_train))
valid_dl = DataLoader(valid_ds,batch_size = 16)


def accuracy(xb,yb):
    out = model(xb)
    preds = torch.argmax(out,dim=1)
    return (preds==yb).float().mean() 

def fit():
    for epoch in range(epochs):
        for xb, yb in train_dl:
            xb = xb.float()
            yb = yb.long()
            print(xb.type())
            print(yb.type())
            out = VGG_16_model(xb)
            loss = loss_function(out,yb)
            loss.backward()
            optim.step()
            optim.zero_grad()
            with torch.no_grad():
                valid_loss = sum(loss_function(model(xb), yb) for xb, yb in valid_dl)   ## sum loss of valid batch
        
        print("epoch: {} - val_loss: {:.2f} - accuracy: {:.2f} - val_acc: {:.2f}".format(
            epoch, 
            (valid_loss / len(valid_dl)).item(),   ## mean() of sum loss
            accuracy(x_train,y_train).item(), 
            accuracy(x_valid,y_valid).item()
        ))

fit()