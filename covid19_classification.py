import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import data_preprocess
import VGG16
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("device: ",device)

x_train,y_train,x_valid,y_valid = data_preprocess.load_data()
VGG_16_model = VGG16.get_model().to(device)

lr = 0.00001
epochs = 20
loss_function = F.cross_entropy
bs = 16
# optim = torch.optim.SGD(VGG_16_model.parameters(),lr = lr)
optim = torch.optim.Adam(VGG_16_model.parameters(),lr = lr)

train_ds = list(zip(x_train,y_train))
train_dl = DataLoader(train_ds,batch_size=16)

valid_ds = list(zip(x_train,y_train))
valid_dl = DataLoader(valid_ds,batch_size = 16)


def accuracy(xb,yb):
    xb = torch.Tensor(xb)
    yb = torch.Tensor(yb)
    xb = xb.float()
    xb.to(device)
    yb = yb.long()
    yb.to(device)
    out = VGG_16_model(xb)
    preds = torch.argmax(out,dim=1)
    return (preds==yb).float().mean() 


def fit():
    print("Optimazer: ",optim)
    for epoch in range(epochs):
        for i,(xb, yb) in enumerate(train_dl):
            xb = xb.float()
            xb = xb.to(device)
            yb = yb.long()
            yb = yb.to(device)
            # print(xb.type())
            out = VGG_16_model(xb)
            loss = loss_function(out,yb)
            if i == 0 :
                train_loss = loss
            else:
                train_loss += loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()
            with torch.no_grad():
                valid_loss = sum(loss_function(VGG_16_model(xb.float().to(device)), yb.long().to(device)) for xb, yb in valid_dl)   ## sum loss of valid batch
        print("epoch: {}- train_loss:{:.2f} - val_loss: {:.2f}".format(
        # print("epoch: {} - val_loss: {:.2f}".format(
            epoch+1,
            (train_loss) / len(train_dl),
            (valid_loss / len(valid_dl)).item()  ## mean() of sum loss
        ))


fit()
# print("final accuracy: ", accuracy(x_valid,y_valid))

# print(x_valid[0].shape)