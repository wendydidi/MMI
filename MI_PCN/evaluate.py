import argparse
import torch
import shapenet_part_loader 
from dataset.dataset import ShapeNet
from MI_demo import AutoEncoder

from utils import PointLoss_test
# from model import AutoEncoder 
from loss import ChamferDistance,CD
import numpy  as np 
import utils
import glob 
import time
import show_points

parser = argparse.ArgumentParser()
parser.add_argument('--partial_root', type=str, default='/home/rico/Workspace/Dataset/shapenetpcn/partial')
parser.add_argument('--gt_root', type=str, default='/home/rico/Workspace/Dataset/shapenetpcn/gt')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--pnum', type=int, default=2048)
parser.add_argument('--crop_point_num', type=int, default=512)

parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')

parser.add_argument('--num_coarse', type=int, default= 512)
parser.add_argument('--num_dense', type=int, default=2048)
# parser.add_argument('--batch_size', type=int, default=16)


args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cd_loss = ChamferDistance()
CD_Loss  = CD()


name2cls = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Guitar': 5, 'Lamp': 6, 'Laptop': 7, 'Motorbike': 8, 'Mug': 9, 'Pistol': 10, 'Skateboard': 11, 'Table': 12}

class_choice = [ 'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Guitar', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol',  'Skateboard', 'Table']
cls2name = dict([(name2cls[key],key) for key in name2cls.keys()])

print(cls2name)

# test_dataset = ShapeNet(partial_path=args.partial_root, gt_path=args.gt_root, split='test')
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_dset = shapenet_part_loader.PartDataset(root='/root/lu/project/ZA/PFNET/PFNet/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/',
                                             classification=True, class_choice=class_choice, npoints=args.pnum, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=int(args.num_workers))





#################################
criterion_PointLoss = PointLoss_test().to(device)

def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 

def Test():

    model_files = glob.glob('./log/lowest_loss.pth')
    # model_files =  [ file for file in model_files if int(file.split("netG")[-1].split(".")[0] )>128 ]
    print(model_files)

    network = AutoEncoder(args)
    # model_files = sorted(model_files)[2:]
    print(model_files)

    whole_outf = open("log/{}_whole_results.txt".format(time.strftime("%Y-%m-%d_%H:%M:%S",time.localtime())),'w')
    # part_outf =  open("log/{}_part_results.txt".format(time.strftime("%Y-%m-%d_%H:%M:%S",time.localtime())),'w')

    for model in model_files:
        print(model)

        network.to(device)
        network.load_state_dict(torch.load(model))   
        network.eval()


        whole_CD_dist1_dist2 = [[[],[],[]] for i in range(len(class_choice))]
        part_CD_dist1_dist2 = [[[],[],[]] for i in range(len(class_choice))]

        errG_min = 100
        n = 0
        CD = 0
        Gt_Pre =0
        Pre_Gt = 0
        IDX = 1

        for i, data in enumerate(test_dataloader, 0):

            real_point, target = data
            partial_input, coarse_gt,dense_gt,real_center,target   = utils.get_batch_data(data,device,args)
            # real_point, target = data
         
            v, y_coarse, y_detail ,LOCAL, GLOBAL, ToT = network(partial_input)

            # v, y_coarse, y_detail  = network(partial_input)

            # print("global:{:.4f} local:{:.4f} ToT:{:.4f}".format(GLOBAL.item(), LOCAL.item(),ToT.item()))

            y_coarse = y_coarse.permute(0, 2, 1)
            y_detail = y_detail.permute(0, 2, 1)
            
            
            # GT_Pre,Pre_GT, loss = CD_Loss(dense_gt, y_detail)
            # GT_Pre = GT_Pre.cpu().detach().numpy()
            # Pre_GT = Pre_GT.cpu().detach().numpy()
            # loss = loss.cpu().detach().numpy()

            loss,GT_Pre,Pre_GT  = criterion_PointLoss(y_detail.squeeze(),dense_gt.squeeze())
            if i%10==0:
                show_points.show_batch(input_list=[dense_gt,partial_input.permute(0, 2, 1),partial_input.permute(0, 2, 1),real_center], output_list=[y_detail,y_coarse], step=i,
                   epoch=0,opt=args)

          
        
            dist  = [GT_Pre,Pre_GT,loss]
            ### whole 
            # dist_all, dist1, dist2 = criterion_PointLoss(torch.squeeze(fake_whole,1),torch.squeeze(real_point,1))
            whole_CD_dist1_dist2 = records(dist ,whole_outf,target,whole_CD_dist1_dist2,i,"whole")

            ###part
            # part_CD_dist1_dist2 = records(fake,real_center,part_outf,target,part_CD_dist1_dist2,i,"part")


        final_print(whole_outf,model,whole_CD_dist1_dist2,"whole")
        # final_print(part_outf,model,part_CD_dist1_dist2,"part")

       
def records(dist,outf,target,CD_dist1_dist2,i, name):

    # dist =  [GT_Pre,Pre_GT,loss]

            
    for index,cl in enumerate(target):
        CD_dist1_dist2[cl][0].append(dist[-1][index])
        CD_dist1_dist2[cl][1].append(dist[0][index] )
        CD_dist1_dist2[cl][2].append(dist[1][index])

    
    if  i%20==0:
        print(name+" [{}/{}] CD:{}  Pre_GT:{} GT_Pre:{}" .format(i,len(test_dset)//args.batch_size,1e3*np.mean(np.array(dist[-1])),1e3*np.mean(np.array(dist[1])),1e3*np.mean(np.array(dist[0]))))

        outf.write( "[{}/{}] CD:{}  Pre_GT:{} GT_Pre:{} \n" .format(i,len(test_dset)//args.batch_size,1e3*np.mean(np.array(dist[-1])),1e3*np.mean(np.array(dist[1])),1e3*np.mean(np.array(dist[0]))))


    return CD_dist1_dist2


def final_print(outf,model,CD_dist1_dist2,name):

    outf.write("\n\n-----------{}------------------\n".format(model))

    CD = []
    Pre_GT = [] 
    GT_Pre = []

    for k,dist in enumerate(CD_dist1_dist2):

        CD+= dist[0]
        Pre_GT += dist[2]
        GT_Pre +=dist[1]

        print(name+ " name:{} CD: {:.4f} Pre_GT:{:.4f} GT_Pre:{:.4f} ".format( cls2name[k],1e3*np.mean(np.array(dist[0])) ,
                                        1e3*np.mean(np.array(dist[2])) ,
                                        1e3*np.mean(np.array(dist[1]))
                                        ))
                    

        outf.write("name:{} CD: {:.4f} Pre_GT:{:.4f} GT_Pre:{:.4f} \n".format( cls2name[k],1e3*np.mean(np.array(dist[0])) ,
                                        1e3*np.mean(np.array(dist[2])) ,
                                        1e3*np.mean(np.array(dist[1]))
                                        )  )

    print(name+ " -------------------Final--------------------\nCD:{}  Pre_GT:{} , GT_Pre:{}".format( 1e3*np.mean(np.array(CD)) ,
                                        1e3*np.mean(np.array(Pre_GT)) ,
                                        1e3*np.mean(np.array(GT_Pre))
                                        )
                                        )
    outf.write(name+ " -------------------Final--------------------\nCD:{}  Pre_GT:{} , GT_Pre:{}\n".format( 1e3*np.mean(np.array(CD)) ,
                                        1e3*np.mean(np.array(Pre_GT)) ,
                                        1e3*np.mean(np.array(GT_Pre))
                                        )
                                        )
      


if __name__=="__main__":

    Test()
    

