import copy
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import shapenet_part_loader
import os
import sys
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils

# import data_utils as d_utils
import torchvision
import numpy as np

from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_depict(data, name, mask):
    # bsize: 数据个数
    # fn: 路径
    # lab2seg: 生成的颜色对应表，即每个种类的每个part类别所对应的颜色，作为输入是为了使得不同结果间保持一致
    # cls: 每个点云数据所对应的类别
    # for j in range(bsize):
    # 读取点云和分割结果数据

    # 归一化
    data[:, :3] = pc_normalize(data[:, :3])
    # 将数据点的part类别对应为lab2seg中设置的颜色
    colormap = [[] for _ in range(len(data))]
    for i in range(len(data)):
        if mask[i] == 0:

            colormap[i] = 'g'
        else:
            colormap[i] = 'r'
    # 设置图片大小
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='3d')
    # 设置视角
    ax.view_init(elev=30, azim=-60)
    # 关闭坐标轴
    plt.axis('off')
    # 设置坐标轴范围
    ax.set_zlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_xlim3d(-1, 1)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colormap, s=20, marker='.')  # , cmap='plasma')
    # plt.show()
    # 保存为.eps文件，也可以是其他类型，注意保存不能和显示同时使用
    plt.savefig('tmp/{}.png'.format(name), dpi=200, bbox_inches='tight', transparent=True)
    plt.close()

    return 'tmp/{}.png'.format(name)


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid[np.newaxis, :]
    m = np.max(np.sqrt(np.sum(np.power(pc, 2), axis=1)), axis=0)
    pc = pc / m[np.newaxis, np.newaxis]
    return pc


def test(point_netG, epoch, test_dataloader,opt):

    for i, data in enumerate(test_dataloader, 0):

        if i%10!=0:
            continue

        print("[{}:{}]".format(len(test_dataloader),i))
        real_point, target = data


        batch_size = real_point.size()[0]
        real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
        input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
        input_cropped1 = input_cropped1.data.copy_(real_point)
        real_point = torch.unsqueeze(real_point, 1)
        input_cropped1 = torch.unsqueeze(input_cropped1, 1)

        input_cropped_partial = torch.FloatTensor(batch_size,1,opt.pnum-opt.crop_point_num,3)
        p_origin = [0, 0, 0]

        if opt.cropmethod == 'random_center':
            choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                        torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
            for m in range(batch_size):
                index = random.sample(choice, 1)
                distance_list = []
                p_center = index[0]
                for n in range(opt.pnum):
                    distance_list.append(distance_squre(real_point[m, 0, n], p_center))
                distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

                for sp in range(opt.crop_point_num):
                    input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
                    real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]

                crop_num_list = []
                for num_ in range(opt.pnum- opt.crop_point_num):
                    crop_num_list.append(distance_order[num_+opt.crop_point_num][0])
                indices = torch.LongTensor(crop_num_list)
                input_cropped_partial[m,0] =torch.index_select(real_point[m,0],0,indices)
        
        input_cropped_partial = input_cropped_partial.squeeze().to(device)
        real_point = real_point.to(device)
        real_center = real_center.to(device)
        input_cropped1 = input_cropped1.to(device)
        ############################
        # (1) data prepare
        ###########################
        # real_center = Variable(real_center,requires_grad=True)
        real_center = torch.squeeze(real_center, 1)

        ## 64  个点
        real_center_key1_idx = utils.farthest_point_sample(real_center, 128, RAN=False)
        real_center_key1 = utils.index_points(real_center, real_center_key1_idx)
        # real_center_key1 =Variable(real_center_key1,requires_grad=True)

        ## 128  个点
        # real_center_key2_idx = utils.farthest_point_sample(real_center, 128, RAN=True)
        # real_center_key2 = utils.index_points(real_center, real_center_key2_idx)
        # real_center_key2 =Variable(real_center_key2,requires_grad=True)

        input_cropped1 = torch.squeeze(input_cropped1, 1).to(device)
        input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1], RAN=True)
        input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
        input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2], RAN=False)
        input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)

        # input_cropped1 = Variable(input_cropped1,requires_grad=True)
        # input_cropped2 = Variable(input_cropped2,requires_grad=True)
        # input_cropped3 = Variable(input_cropped3,requires_grad=True)
        input_cropped2 = input_cropped2.to(device)
        input_cropped3 = input_cropped3.to(device)
        input_cropped = [input_cropped1, input_cropped2, input_cropped3]
        point_netG = point_netG.eval()
        fake_center1, fake = point_netG(input_cropped)

        show_batch(input_list=[real_point, input_cropped_partial,input_cropped1], output_list=[fake,  fake_center1], step=i,
                   epoch=epoch,opt=opt)
        point_netG = point_netG.train()


def show_batch(input_list, output_list, step, epoch,opt):
    """
    real_point :gt
    input_cropped1:input

    fake
    fake_center1
    fake_center2

    input_list: [real_point,input_cropped1]
    output_list: [fake,key2,key1]
    """
    gt = input_list[0].squeeze()
    input_cropped_partial = input_list[1].squeeze()
    crop_input = input_list[-2].squeeze()
    real_center = input_list[-1].squeeze()

    key3 = output_list[0].squeeze()
    # key2 = output_list[1].squeeze()
    key1 = output_list[-1].squeeze()

    # print(crop_input.shape)
    N =  input_cropped_partial.shape[1]

    print(input_cropped_partial.shape,crop_input.shape,key1.shape,key3.shape,real_center.shape)

    path_list = []

    for i in range(gt.shape[0]):
        # mask[1000:] = 1
        mask = np.zeros(gt[i].shape[0])
        GT_path = data_depict(gt[i, ...].cpu().detach().numpy(), "{}_0_GT".format( i), mask)


        input = torch.cat([crop_input.squeeze()[i, ...],real_center.squeeze()[i, ...]], dim=0)
        mask = np.zeros(input.shape[0])
        # mask = np.zeros(crop_input[i].shape[0])
        # print(input.shape,crop_input[i].shape)
        mask[crop_input[i].shape[0]:] = 1
        crop_input_path = data_depict(crop_input[i].squeeze().cpu().detach().numpy(), "{}_1_cropinput".format( i), mask)
        # print(input_cropped1.shape,fake.shape)

        input = key3.squeeze()[i, ...]
        mask = np.ones(input.shape[0])
        # mask[N:] = 1
        key3_path = data_depict(input.cpu().detach().numpy(), "{}_3_key3".format( i), mask)



        input =  key1.squeeze()[i, ...]
        mask = np.ones(input.shape[0])
       
        key1_path = data_depict(input.cpu().detach().numpy(), "{}_2_key1".format( i), mask)

        path_list.append(GT_path)
        path_list.append(crop_input_path)
        path_list.append(key3_path)
       
        path_list.append(key1_path)

    sorted(path_list)


    imgs = []
    for img_path in path_list:
        im = Image.open(img_path)
        imgs.append(np.expand_dims(np.array(im)[:, :, :-1], 0))
    imgs = np.concatenate(imgs, axis=0)

    imgs = imgs.astype(np.float32)
  
    torchvision.utils.save_image(torch.from_numpy(imgs).permute(0, 3, 2, 1),
                                 'test_results/{}_{}_test.png'.format(epoch, step),4, normalize=True, padding=2)

    # GT input  512  128



def compute_radius():

    class_choice = [ 'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Guitar', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol',  'Skateboard', 'Table']


    dset = shapenet_part_loader.PartDataset(root='../../PFNET/PFNet/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/',
                                                classification=True, class_choice=class_choice, npoints=2048, split='train')
    dataloader = torch.utils.data.DataLoader(dset, batch_size=16,
                                                shuffle=True, num_workers=int(2))

    Radius = []

    for i, data in enumerate(dataloader, 0):

        # print("[{}:{}]".format(len(test_dataloader),i))
        real_point, target = data


        batch_size = real_point.size()[0]
        real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
        input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
        input_cropped1 = input_cropped1.data.copy_(real_point)
        real_point = torch.unsqueeze(real_point, 1)
        input_cropped1 = torch.unsqueeze(input_cropped1, 1)

        input_cropped_partial = torch.FloatTensor(batch_size,1,opt.pnum-opt.crop_point_num,3)
        p_origin = [0, 0, 0]

        if opt.cropmethod == 'random_center':
            choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                        torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
            for m in range(batch_size):
                index = random.sample(choice, 1)
                distance_list = []
                p_center = index[0]
                for n in range(opt.pnum):
                    distance_list.append(distance_squre(real_point[m, 0, n], p_center))
                distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

                for sp in range(opt.crop_point_num):
                    input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
                    real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]

                crop_num_list = []
                for num_ in range(opt.pnum- opt.crop_point_num):
                    crop_num_list.append(distance_order[num_+opt.crop_point_num][0])
                indices = torch.LongTensor(crop_num_list)
                input_cropped_partial[m,0] =torch.index_select(real_point[m,0],0,indices)
        
        input_cropped_partial = input_cropped_partial.squeeze().to(device)
        real_point = real_point.to(device)
        real_center = real_center.to(device)
       
        real_center = torch.squeeze(real_center, 1)

        avg_radius = utils.get_K_dist(real_center)
        Radius.append(avg_radius)

        if i%10==0:
            r = torch.mean(torch.Tensor(Radius))
            print("[{}\{}] avarage r:{:.4f}".format(len(dataloader),i,r))



if __name__ == '__main__':
    # data= np.loadtxt("test_one/crop_ours_txt.txt",delimiter=',').astype(np.float32)
    # test_dset = shapenet_part_loader.PartDataset(root='../PF_dataset/shapenetcore_partanno_segmentation_benchmark_v0/',
    #                                              classification=True, class_choice=None, npoints=2048, split='test')
    # test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=8,
    #                                               shuffle=True, num_workers=int(2))

    # point_netG = _netG(opt.num_scales, opt.each_scales_size, opt.point_scales_list, opt.crop_point_num)
    # point_netG = point_netG.to(device)
    # test(point_netG, 0, test_dataloader,opt)


    compute_radius()


