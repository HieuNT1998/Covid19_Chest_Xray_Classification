import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import data_preprocess
import VGG16
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

print(torch.cuda.is_available())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


