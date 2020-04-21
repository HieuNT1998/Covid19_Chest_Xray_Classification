import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import data_preprocess
import VGG16
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

d = [i for i in range(1000)]
c = [i for i in range(1000)]

e = list(zip(d,c))
e = DataLoader(e,batch_size=2)

for d,c in e:
    print("d: ", d)
    print("c: ", c)


