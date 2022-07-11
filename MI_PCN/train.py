import argparse

import torch
import torch.optim as optim
import shapenet_part_loader 

from dataset.dataset import ShapeNet
# from MI_demo import AutoEncoder
from model import AutoEncoder
from loss import ChamferDistance, EarthMoverDistance
import utils
import glob  


parser = argparse.ArgumentParser()
parser.add_argument('--partial_root', type=str, default='/home/rico/Workspace/Dataset/shapenetpcn/partial')
parser.add_argument('--gt_root', type=str, default='/home/rico/Workspace/Dataset/shapenetpcn/gt')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--num_input', type=int, default=2048)

parser.add_argument('--pnum', type=int, default=2048)
parser.add_argument('--crop_point_num', type=int, default=512)

parser.add_argument('--dropout', type=float, default=0.1)
# parser.add_argument('--cropmethod', type=bool, default=True)

parser.add_argument('--num_coarse', type=int, default= 512)
parser.add_argument('--num_dense', type=int, default=2048)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')

parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--loss_d1', type=str, default='cd')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=151)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--log_dir', type=str, default='log')
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cd_loss = ChamferDistance()
emd_loss = EarthMoverDistance()
loss_d1 = cd_loss if args.loss_d1 == 'cd' else emd_loss
loss_d2 = cd_loss

# train_dataset = ShapeNet(partial_path=args.partial_root, gt_path=args.gt_root, split='train')
# val_dataset = ShapeNet(partial_path=args.partial_root, gt_path=args.gt_root, split='val')
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

###################################################

# lu/project/PFNET/PFNet/dataset

class_choice = [ 'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Guitar', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol',  'Skateboard', 'Table']

dset = shapenet_part_loader.PartDataset(root='/root/lu/project/ZA/PFNET/PFNet/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/',
                                        classification=True, class_choice=class_choice, npoints=args.pnum, split='train')
assert dset
train_dataloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.num_workers))

test_dset = shapenet_part_loader.PartDataset(root='/root/lu/project/ZA/PFNET/PFNet/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/',
                                             classification=True, class_choice=class_choice, npoints=args.pnum, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=int(args.num_workers))


#####################################################


network = AutoEncoder(args)
if args.model is not None:
    print('Loaded trained model from {}.'.format(args.model))
    network.load_state_dict(torch.load(args.model))
else:
    print('Begin training new model.')
network.to(device)
optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

max_iter = int(len(dset) / args.batch_size + 0.5)
minimum_loss = 1e4
best_epoch = 0

for epoch in range(1, args.epochs + 1):
    # training
    network.train()
    total_loss, iter_count = 0, 0
    for i, data in enumerate(train_dataloader, 1):

        #########
        
        partial_input, coarse_gt,dense_gt,real_center,target   = utils.get_batch_data(data,device,args)

        optimizer.zero_grad()

        # v, y_coarse, y_detail ,LOCAL, GLOBAL, ToT = network(partial_input)
        v, y_coarse, y_detail  = network(partial_input)
        # print("global:{:.4f} local:{:.4f} ToT:{:.4f}".format(GLOBAL.item(), LOCAL.item(),ToT.item()))

        y_coarse = y_coarse.permute(0, 2, 1)
        y_detail = y_detail.permute(0, 2, 1)

        #### loss
        loss = loss_d1(coarse_gt, y_coarse) + args.alpha * loss_d2(dense_gt, y_detail)
        loss.backward()
        optimizer.step()
        
        iter_count += 1
        total_loss += loss.item()
        
        if i % 100 == 0:
            print("Training epoch {}/{}, iteration {}/{}: loss is {:.4f}".format(epoch, args.epochs, i, max_iter, loss.item()))
    scheduler.step()

    print("\033[31mTraining epoch {}/{}: avg loss = {}\033[0m".format(epoch, args.epochs, total_loss / iter_count))

    # evaluation
    network.eval()
    with torch.no_grad():
        total_loss, iter_count = 0, 0
        for i, data in enumerate(test_dataloader, 1):


            """
            
            """
            partial_input, coarse_gt,dense_gt,real_center, target   = utils.get_batch_data(data,device,args)
            
            # partial_input = partial_input.to(device)
            # coarse_gt = coarse_gt.to(device)
            # dense_gt = dense_gt.to(device)
            # partial_input = partial_input.permute(0, 2, 1)
            
            # v, y_coarse, y_detail ,LOCAL, GLOBAL, ToT = network(partial_input)
            v, y_coarse, y_detail  = network(partial_input)
            # print("global:{:.4f} local:{:.4f} ToT:{:.4f}".format(GLOBAL.item(), LOCAL.item(),ToT.item()))

            y_coarse = y_coarse.permute(0, 2, 1)
            y_detail = y_detail.permute(0, 2, 1)

            loss = loss_d1(coarse_gt, y_coarse) + args.alpha * loss_d2(dense_gt, y_detail)
            total_loss += loss.item()
            iter_count += 1

        mean_loss = total_loss / iter_count
        print("\033[31mValidation epoch {}/{}, loss is {}\033[0m".format(epoch, args.epochs, mean_loss))

        # records the best model and epoch
        if mean_loss < minimum_loss:
            best_epoch = epoch
            minimum_loss = mean_loss
            torch.save(network.state_dict(), args.log_dir + '/lowest_loss_{}.pth'.format(epoch ))

    print("\033[31mBest model (lowest loss) in epoch {}\033[0m".format(best_epoch))
