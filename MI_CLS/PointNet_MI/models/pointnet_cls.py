import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.pointnet import PointNetEncoder, PointNetEncoderNew, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)  # x - (batch_size, 1024) # trans_feat - (batch_size, 64, n)
        x = F.relu(self.bn1(self.fc1(x)))  # -> (batch_size, 512)
        x = F.relu(self.bn2(self.dropout(self.fc2(x)))) # (batch_size, 256)
        x = self.fc3(x) # batch_size, k
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


class get_mi_model(nn.Module):
    # def __init__(self, k=40, normal_channel=True):
    def __init__(self, k=40):
        super(get_mi_model, self).__init__()
        # if normal_channel:
        #     channel = 6
        # else:
        #     channel = 3
        channel = 3
        self.feat = PointNetEncoderNew(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat, x_local = self.feat(x)  # x - (batch_size, 1024) # trans_feat - (batch_size, 64, 64)
                                                      # trans - (b, 3, 3) # x_local - (b, 64, n)
        #########################################################
        batch_size = x.size(0)
        num_points = x_local.size(2)
        x_global = F.adaptive_max_pool1d(x_local, 1).view(batch_size, -1)  # ->(batch_size, 64)

        x_local_prime = x_local[torch.randperm(x_local.size(0))] # ->(batchsize, 64, num_points)
        x_global_prime = x_global[torch.randperm(x_global.size(0))] # ->(batchsize, 64)
        #########################################################
        x = F.relu(self.bn1(self.fc1(x)))  # -> (batch_size, 512)
        x = F.relu(self.bn2(self.dropout(self.fc2(x)))) # (batch_size, 256)
        x = F.relu(self.bn3(self.dropout(self.fc3(x)))) # (batch_size, 64)
        # x = F.relu(self.bn3(self.fc3(x)))
        # print(x.size())
        c = x                                              # (batch_size, 64)
        c_exp = c.unsqueeze(2)
        c_exp = c_exp.expand(-1, -1, num_points)                   # (b, 64, n)
        # x = self.dropout(x)
        # x = F.relu(self.bn3(self.dropout(self.fc3(x))))   
        x = self.fc4(x)                                     # (b, k)
        x = F.log_softmax(x, dim=1)
        return x_local, x_local_prime, x_global, x_global_prime, c, c_exp, x, trans_feat
        # x_local = trans_feat ->  (b, 64, n)
        # x_local_prime -> (batch_size, 64, n)
        # x_global, x_global_prime -> (batch_size, 64)
        # c -> (batch_size, 64)
        # c_exp -> (batch_size, 64, n)
        # x - (batch_size, k)


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)  # a value
        # print(mat_diff_loss)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss # a value


class GlobalinfolossNet(nn.Module):
    def __init__(self):
        super(GlobalinfolossNet, self).__init__()
        self.c1 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
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
## each pair xx:  b*(128+128)*1024
class LocalinfolossNet(nn.Module):
    def __init__(self):
        super(LocalinfolossNet, self).__init__()
        self.conv1 = nn.Conv1d(128, 64, kernel_size=1, bias=False)
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
        super(DeepMILoss, self).__init__()

        self.global_d = GlobalinfolossNet()
        self.local_d = LocalinfolossNet()
        
   
    def forward(self, x_global, x_global_prime, x_local, x_local_prime, c, c_p):
        # x_local: (batch_size, 64, num_points) 
        # x_local_prime: (batch_size, 64, num_points)
        # x_global: (batch_size, 64)  
        # x_global_prime: (batch_size, 64)
        # c: (batch_size, 64, num_points) --- c3
        # c_p: (batch_size, 64) --- c2

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
