import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.optim as optim
from torchvision.ops import *
from torchvision.models import *
from torchsummary import summary
from matplotlib import pyplot as plt
# import bitsandbytes as bnb
import numpy as np
import pandas as pd
import scipy.stats
import os
import time
import copy
import random
from tqdm import tqdm, trange

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:0')
print(torch.cuda.is_available())

# preprocessing
from preprocess import *
ExpressionData = pd.read_csv('../Benchmark Dataset/Non-Specific Dataset/hESC/TFs+1000/BL--ExpressionData.csv', index_col=0, engine='c')
network = pd.read_csv('../Benchmark Dataset/Non-Specific Dataset/hESC/TFs+1000/Label.csv', index_col=0, engine='c').values
tfs_raw = pd.read_csv('../Benchmark Dataset/Non-Specific Dataset/hESC/TFs+1000/TF.csv', index_col=0, engine='c')['index'].values.tolist()
targets_raw = pd.read_csv('../Benchmark Dataset/Non-Specific Dataset/hESC/TFs+1000/Target.csv', index_col=0, engine='c')['index'].values.tolist()

batch_size = 512
patch_size = 8
gene_pair, all_exp, labels = processing(tfs_raw, targets_raw, ExpressionData.values, network)
data = Feeder(np.array(all_exp), labels, patch_size, 'pretrain')
loader = DataLoader(data, batch_size=batch_size, shuffle=True)
labels_unique, counts = np.unique(labels, return_counts=True)
class_weight = [sum(counts)/i for i in counts]

# load model
from models import*
model = MAE(seq_len=data[0][0].shape[1], 
            channels=2, 
            patch_size=patch_size,
            cls_token=False,)

# define optimizer
opt = optim.AdamW(model.parameters())
scaler = amp.GradScaler()

# create folder to save model weights
os.makedirs('./models', exist_ok=True)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
# start training
num_epochs = 200
encoder, loss_history = Training(model.to(device), num_epochs=num_epochs)

# plot loss history
plt.title('Loss History')
plt.plot(range(1, num_epochs+1), loss_history, label='train')
plt.ylabel('Loss')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()

# downstream
from lincls import *
train_index, val_index, test_index = split_datasets(labels)
exp_train = np.array(all_exp)[np.array(train_index)]
y_train = np.array(labels)[np.array(train_index)]
exp_val = np.array(all_exp)[np.array(val_index)]
y_val = np.array(labels)[np.array(val_index)]
exp_test = np.array(all_exp)[np.array(test_index)]
y_test = np.array(labels)[np.array(test_index)]

train_data = Feeder(np.array(exp_train), y_train, patch_size, 'lincls')
val_data = Feeder(np.array(exp_val), y_val, patch_size, 'lincls')
test_data = Feeder(np.array(exp_test), y_test, patch_size, 'lincls')
train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# without finetune (frozen backbone params)
Metric = downstream(model=LinearClassifier(backbone=copy.deepcopy(encoder), finetune=False).to(device), linear_epoch=100)
print(Metric)