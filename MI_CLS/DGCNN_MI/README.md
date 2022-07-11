## DGCNN* + MMI

This is the code for the point cloud classification task and shape retrieval in our work - Maximizing Mutual Information for Learning on Point Clouds.

This is implementation of DGCNN* + MMI in pytorch. This code is based on [DGCNN-pytorch](https://github.com/AnTao97/dgcnn.pytorch.git).

## environments
- Python 3.7
- PyTorch 1.2
- CUDA 10.0
  
glob, h5py, sklearn

## Data 
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

## train the model
```
python main_clswiothmi.py
```

## test the model

``` 
python main_clswiothmi.py --eval=True --model_path=pretrained_model/model.t7
```
## retrieve the shape
```
python main_retrieval.py --eval=True --model_path=pretrained_model/model.t7
```