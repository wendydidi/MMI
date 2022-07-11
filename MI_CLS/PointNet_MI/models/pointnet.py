import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) # (batch_size, channel, num_points) -> (batch_size, 64, num_points)
        x = F.relu(self.bn2(self.conv2(x))) # (batch_size, 64, num_points) -> (batch_size, 128, num_points)
        x = F.relu(self.bn3(self.conv3(x))) # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0] # (batch_size, )
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) 

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x # (batch_size, 3, 3)


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)   # (batchsize, 256)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x  #(batchsize, 64, 64)


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size() # (batch_size, 3, num_points)
        trans = self.stn(x) # (batch_size, 3, 3)
        x = x.transpose(2, 1)
        if D >3 :
            x, feature = x.split(3,dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x))) # (batch_size, 64, n)

        if self.feature_transform:
            trans_feat = self.fstn(x)  # (batch_size, 64, 64)
            x = x.transpose(2, 1) 
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1) # (batch_size, 64, n)
        else:
            trans_feat = None

        pointfeat = x # (batch_size, 64, n)
        x = F.relu(self.bn2(self.conv2(x))) # (batch_size, 128, n)
        x = self.bn3(self.conv3(x)) # (batch_size, 1024, n)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024) # (batch_size, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N) # (batch_size, 1024, n)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
            ## #1 global_feature: (batch_size, 1024)
            ## #1 else: (batch_size, 1088, n)
            #2 -- (batch_size, 3, 3)
            #3 -- (batch_size, 64, n)

class PointNetEncoderNew(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoderNew, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size() # (batch_size, 3, num_points)
        trans = self.stn(x) # (batch_size, 3, 3)
        x = x.transpose(2, 1)
        if D >3 :
            x, feature = x.split(3,dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x))) # (batch_size, 64, n)

        ########################################################
        x = F.relu(self.bn2(self.conv2(x))) # (batch_size, 64, n)
        x_local = x # (b, 64, n)
        ########################################################

        if self.feature_transform:
            trans_feat = self.fstn(x)  # (batch_size, 64, 64)
            x = x.transpose(2, 1) 
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1) # (batch_size, 64, n)
        else:
            trans_feat = None

        pointfeat = x # (batch_size, 64, n)
        x = F.relu(self.bn3(self.conv3(x))) # (batch_size, 128, n)
        x = self.bn4(self.conv4(x)) # (batch_size, 1024, n)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024) # (batch_size, 1024)
        if self.global_feat:
            return x, trans, trans_feat, x_local
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N) # (batch_size, 1024, n)
            return torch.cat([x, pointfeat], 1), trans, trans_feat, x_local
            ## #1 global_feature: (batch_size, 1024)
            ## #1 else: (batch_size, 1088, n)
            #2 -- (batch_size, 3, 3)
            #3 -- (batch_size, 64, n)


def feature_transform_reguliarzer(trans):
    # trans: b*64*64
    d = trans.size()[1]   # d =64 
    I = torch.eye(d)[None, :, :]  # I: 64, 64
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss
    # trans: b* c *n
