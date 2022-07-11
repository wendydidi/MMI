from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
import numpy as np
import sys
from data import ModelNet40
from torch.utils.data import DataLoader
from models.pointnet_cls import get_model, get_mi_model, get_loss, DeepMILoss
from sklearn.metrics.pairwise import cosine_similarity
import time 
# from torch.utils.tensorboard import SummaryWriter   
import datetime


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def train(args, io):
    ## load dataset
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    ## try to load model ##
    model = get_mi_model().to(device)
    print(str(model))
    
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    ## choose opt $ scheduler ##
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.7)

    ## initialization ##
    cls_loss = get_loss()
    MILoss = DeepMILoss().to(device)
    best_test_acc = 0

    for epoch in range(args.epochs):

        start = time.time()

        ###### train #######
        train_loss = 0.0
        train_clsloss = 0.0
        train_miloss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []


        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            ##########################################
            x_local, x_local_prime, x_global, x_global_prime, c, c_exp, x, trans_feat = model(data)
            pred = x # b*40
            clsloss = cls_loss(pred, label, trans_feat)
            lmiloss, gmiloss, tmiloss = MILoss(x_global, x_global_prime, x_local, x_local_prime, c_exp, c)
            miloss = tmiloss
            loss = clsloss + miloss
            ########################################################
            # preds = pred.data.max(1)[1]
            preds = pred.max(dim=1)[1]
            loss.backward()
            opt.step()
            count += batch_size
            train_loss += float(loss.item() * batch_size)
            train_clsloss += float(clsloss.item() * batch_size)
            train_miloss += float(miloss.item() * batch_size)

            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_avg_per_class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        
        end = time.time()
        t = end - start

        outstr = 'Train %d, train acc: %.6f, train avg acc: %.6f, training time: %.6f' % (epoch, train_acc,
                                                                                 train_avg_per_class_acc, t)
        io.cprint(outstr)

        # writer.add_scalar('Training_time', t, epoch)
        # writer.add_scalar('Train_loss', train_loss, epoch)
        # writer.add_scalar('Train_Clsloss', train_clsloss, epoch)
        # writer.add_scalar('Train_MIloss', train_miloss, epoch)
        # writer.add_scalar('Train_acc', train_acc, epoch)
        # writer.add_scalar('Train_avg_per_class_acc', train_avg_per_class_acc, epoch)


        ### test during train ###
        test_loss = 0.0
        test_clsloss = 0.0
        test_miloss = 0.0
        count = 0.0
        model.eval()

        test_pred = []
        test_true = []

        for data, label in test_loader:    
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            ##############       
            x_local, x_local_prime, x_global, x_global_prime, c, c_exp, x, trans_feat = model(data)
            pred = x
            clsloss = cls_loss(pred, label, trans_feat)
            lmiloss, gmiloss, tmiloss = MILoss(x_global, x_global_prime, x_local, x_local_prime, c_exp, c)
            miloss = tmiloss
            loss = clsloss + miloss
            ###############
            preds = pred.max(dim=1)[1]

            count += batch_size
            test_loss += float(loss.item() * batch_size)
            test_clsloss += float(clsloss.item() * batch_size)
            test_miloss += float(miloss.item() * batch_size)

            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, test acc: %.6f, test avg acc: %.6f' % (epoch, test_acc, test_avg_per_class_acc)
        io.cprint(outstr)

        # writer.add_scalar('Test_loss', test_loss, epoch)
        # writer.add_scalar('Test_Clsloss', test_clsloss, epoch)
        # writer.add_scalar('Test_MIloss', test_miloss, epoch)
        # writer.add_scalar('Test_acc', test_acc, epoch)
        # writer.add_scalar('Test_avg_per_class_acc', test_avg_per_class_acc, epoch)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            print('Save model...')
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)

def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = get_mi_model().to(device)
    print(str(model))
    model = nn.DataParallel(model)
    # load the trained model
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        x_local, x_local_prime, x_global, x_global_prime, c, c_exp, x, trans_feat = model(data)
        pred = x
        preds = pred.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

def retrieval(args):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = get_mi_model().to(device)
    # model = get_model().to(device)
    print(str(model))
    model = nn.DataParallel(model)
    # load the trained model
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    Label = []
    shapecode = []
    Data = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        x_local, x_local_prime, x_global, x_global_prime, c, c_exp, x, trans_feat = model(data)
        # c -- shape code
        Label.append(label.cpu().numpy())
        shapecode.append(c.detach().cpu().numpy())
        Data.append(data.detach().cpu().numpy())
    
    Data = np.concatenate(Data)
    Label = np.concatenate(Label)
    # Label = np.array(np.concatenate(Label))
    shapecode = np.concatenate(shapecode)
    # print(len(Label))  # 2468
    # print(len(shapecode[1])) # 2468 64
    cosinedistance = cosine_similarity(shapecode) # distance matrix : dim n*n
    top10 = np.zeros((shapecode.shape[0], 10), dtype=np.int) # the index of the samples with highest similarity
    top10_label = np.zeros((shapecode.shape[0], 10), dtype=np.int) # the corresponding label of top10 samples
    
    # acc = 0
    # for i in range(shapecode.shape[0]):
    #     top10[i] = np.argsort(cosinedistance[i])[-11:-1]
    #     top10_label[i] = Label[top10[i]]
    #     acc += (top10_label[i]==Label[i]).sum()
    #     print('Object:', i, 'Label:', Label[i], 'Retrieval result - Top10:', top10_label[i])
    #     # print('retrieval Top10 shape:', top10_label[i])

    # totacc = acc/shapecode.shape[0]/10
    # print('the accurcy of shape retrieval is ', totacc)

    num_classes = np.unique(Label)
    acc_list = []
    for num in num_classes:
        acc = 0
        num_class = 0
        for i in range(shapecode.shape[0]):
            if Label[i]==num:
                num_class += 1
                top10[i] = np.argsort(cosinedistance[i])[-11:-1]
                top10_label[i] = Label[top10[i]]
                acc += (top10_label[i]==Label[i]).sum()/10

        acc_list.append(acc/num_class)
    acc = np.mean(acc_list)
    print("mAP:", acc)
    print("End retrievaling")



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N', choices=['modelnet40'])    
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training [default: 32]')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epochs',  default=300, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--cuda', type=bool, default=True, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False, help='evaluate the model')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # _init_()

    # io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    # io.cprint(str(args))

    torch.manual_seed(args.seed)
    print('Using GPU:' + str(torch.cuda.current_device()) + 'from' + str(torch.cuda.device_count()) + ' devices' )
    # if args.cuda:
    #     io.cprint(
    #         'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    #     torch.cuda.manual_seed(args.seed)
    # else:
    #     io.cprint('Using CPU')
    
    retrieval(args)
