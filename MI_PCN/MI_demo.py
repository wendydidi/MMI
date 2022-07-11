#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MI

class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()

        
        # first shared mlp
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)


        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        # second shared mlp
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)

        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)

        ###

        self.conv5 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        
        # self.linear4 = nn.Linear(64, output_channels)

    
    def forward(self, x):
        n = x.size()[2]
        batch_size = x.size()[0]

        # first shared mlp
        x = F.relu(self.bn1(self.conv1(x)))           # (B, 128, N)
        f = self.bn2(self.conv2(x))                   # (B, 256, N)

        x_local = f # (batch_size, 256, num_points)
        x_global = F.adaptive_max_pool1d(x_local, 1).view(batch_size, -1)  # ->(B, 256)
        x_local_prime = x_local[torch.randperm(x_local.size(0))] # ->(B, 256, N)
        x_global_prime = x_global[torch.randperm(x_global.size(0))] # ->(B, 256)
        
        # point-wise maxpool
        g = torch.max(f, dim=2, keepdim=True)[0]      # (B, 256, 1)
        # expand and concat
        x = torch.cat([g.repeat(1, 1, n), f], dim=1)  # (B, 512, N)

        # second shared mlp
        x = F.relu(self.bn3(self.conv3(x)))           # (B, 512, N)
        x = self.bn4(self.conv4(x))                   # (B, 1024, N)

        #####################
        #### 
        c, c_exp = self.get_deep_features(x,batch_size,n)
        #####################
        
        # point-wise maxpool
        v = torch.max(x, dim=-1)[0]                   # (B, 1024)
        
        return v,  x_global, x_global_prime, x_local, x_local_prime, c, c_exp
    

    def get_deep_features(self,x,batch_size, num_points):

        x = self.conv5(x)                          # (batch_size, 1024, num_points) -> (batch_size, 512, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # -> (batch_size, 512)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) # -> (batch_size, 512)
        x = torch.cat((x1, x2), 1)                            # -> (batch_size, 1024)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
       
        c = x # (batch_size, 256)
        c_exp = c.unsqueeze(2)
        c_exp = c_exp.expand(-1, -1, num_points)
    
        return c, c_exp



class Decoder(nn.Module):
    def __init__(self, num_coarse=512, num_dense=2048):
        super(Decoder, self).__init__()

        self.num_coarse = num_coarse
        
        # fully connected layers
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 3 * num_coarse)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

        # shared mlp
        self.conv1 = nn.Conv1d(3+2+1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)


        self.conv3 = nn.Conv1d(256, 3, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)

        # 2D grid
        grids = np.meshgrid(np.linspace(-0.04, 0.04, 2, dtype=np.float32),
                            np.linspace(-0.04, 0.04, 2, dtype=np.float32))                               # (2, 4, 44)
        self.grids = torch.Tensor(grids).view(2, -1)  # (2, 4, 4) -> (2, 4)

    
    def forward(self, x):
        b = x.size()[0]
        # global features
        v = x  # (B, 1024)

        # fully connected layers to generate the coarse output
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        y_coarse = x.view(-1, 3, self.num_coarse)  # (B, 3, 512)

        repeated_centers = y_coarse.unsqueeze(3).repeat(1, 1, 1, 4).view(b, 3, -1)  # (B, 3, 16x1024)
        repeated_v = v.unsqueeze(2).repeat(1, 1, 4 * self.num_coarse)               # (B, 1024, 16x1024)
        grids = self.grids.to(x.device)  # (2, 16)
        grids = grids.unsqueeze(0).repeat(b, 1, self.num_coarse)                     # (B, 2, 16x1024)

        x = torch.cat([repeated_v, grids, repeated_centers], dim=1)                  # (B, 2+3+1024, 16x1024)
        x = F.relu(self.bn3(self.conv1(x)))
        x = F.relu(self.bn4(self.conv2(x)))
        x = self.conv3(x)                # (B, 3, 16x1024)
        y_detail = x + repeated_centers  # (B, 3, 16x1024)

        return y_coarse, y_detail


class AutoEncoder(nn.Module):
    def __init__(self,args):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(args)
        self.decoder = Decoder()

        self.MI = MI.DeepMILoss()
    
    def forward(self, x):
        v,  x_global, x_global_prime, x_local, x_local_prime, c, c_exp = self.encoder(x)

        y_coarse, y_detail = self.decoder(v)

        LOCAL, GLOBAL, ToT =  self.MI(x_global, x_global_prime, x_local, x_local_prime,  c_exp,c)

        return v, y_coarse, y_detail, LOCAL, GLOBAL, ToT


if __name__ == "__main__":
    pcs = torch.rand(16, 3, 2048)
    encoder = Encoder()
    v = encoder(pcs)
    print(v.size())

    decoder = Decoder()
    decoder(v)
    y_c, y_d = decoder(v)
    print(y_c.size(), y_d.size())

    ae = AutoEncoder()
    v, y_coarse, y_detail = ae(pcs)
    print(v.size(), y_coarse.size(), y_detail.size())





