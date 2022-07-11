## PointNet* + MMI

This is the code for the point cloud classification task and shape retrieval in our work - Maximizing Mutual Information for Learning on Point Clouds.

This is implementation of PointNet* + MMI in pytorch. This code is based on [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet-pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git).

## environments
- Python 3.7
- PyTorch 1.2
- CUDA 10.0

## Environments
Ubuntu 18
Python 3.6
Pytorch 1.2

## Data 
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

## train the model
```
python train_test.py 
```
## test the model
```
python train_test.py --eval=True --model_path=pretrained_model/trained_model/model.t7
```
## retrieve the shape

python main_retrieval.py --model_path=pretrained_model/trained_model/model.t7

