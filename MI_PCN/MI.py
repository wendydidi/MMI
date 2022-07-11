
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# input size: (b, 64)
class GlobalinfolossNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(256*2, 128, kernel_size=1, bias=False)
        self.c2 = nn.Conv1d(128, 64, kernel_size=1, bias=False)
        self.c3 = nn.Conv1d(64, 32, kernel_size=1, bias=False)
        self.l0 = nn.Linear(32, 1)
    
    def forward(self, x_global, c):
        # input size: (b, 64)
        # x_global = b*64   c = b*64
        xx = torch.cat((x_global, c), dim = 1)  # -> (b, 128)
        h = xx.unsqueeze(dim=2) # -> (b, 128, 1)
        h = F.relu(self.c1(h)) # -> (b, 128, 1)
        h = F.relu(self.c2(h)) # -> (b, 64, 1)
        h = F.relu(self.c3(h)) # -> (b, 32, 1)
        h = h.view(h.shape[0], -1) # (b, 32)

        return self.l0(h)  # b*1


## repeat shape code before computing the loss
## input local feature b*128*1024 ( batch_size * features * num_points )
## input repeated shape code b*128*1024
## each pair:  b*(128+128)*1024
class LocalinfolossNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(256*2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 1, kernel_size=1, bias=False)
    
    def forward(self, x_local, c):
        # x_local: b* 64* n
        # c : b* 64* n
        xx = torch.cat((x_local, c), dim=1) # -> (b, 128, num_points)
        h = F.relu(self.conv1(xx))  # (b, 128, num_points) -> (b, 64, num_points)
        h = F.relu(self.conv2(h)) #(b, 64, num_points) -> (b, 64, num_points)
        h = F.relu(self.conv3(h))  # (b, 64, num_points) -> (b, 1, num_points)
        h = h.view(h.shape[0], -1) # (b, num_points)
        return h # (b, num_points)


class DeepMILoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.global_d = GlobalinfolossNet()
        self.local_d = LocalinfolossNet()
        
   
    def forward(self, x_global, x_global_prime, x_local, x_local_prime, c, c_p):
        # x_local: (batch_size, 64, num_points) 
        # x_local_prime: (batch_size, 64, num_points)
        # x_global: (batch_size, 64)  
        # x_global_prime: (batch_size, 64)
        # c: (batch_size, 64, num_points) --- c3
        # c_p: (batch_size, 64) --- c2

        # print(x_global.shape,x_global_prime.shape,x_local.shape,x_local_prime.shape,c.shape,c_p.shape)
        ###### local loss ############

        Ej = -F.softplus(-self.local_d(c, x_local)).mean() # positive pairs
        Em = F.softplus(self.local_d(c, x_local_prime)).mean() # negetive pairs
        LOCAL = (Em - Ej) * 0.5
        

        ###### global loss ###########
        Ej = -F.softplus(-self.global_d(c_p, x_global)).mean() # positive pairs
        Em = F.softplus(self.global_d(c_p, x_global_prime)).mean() # negetive pairs
        GLOBAL = (Em - Ej) * 0.5

        ######### combine local and global loss ###############
        ToT = LOCAL + GLOBAL

        return LOCAL, GLOBAL, ToT # tensor, a value