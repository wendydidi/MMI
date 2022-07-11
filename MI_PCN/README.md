# PCN* + MMI

This is the code for the point cloud completion task in our work - Maximizing Mutual Information for Learning on Point Clouds.

This is implementation of PCN*+MMI in pytorch. This code is based on [PCN-tensorflow](https://github.com/wentaoyuan/pcn.git) and [PCN-pytorch](https://github.com/qinglew/PCN-PyTorch.git).

This code will be released to the public after the paper received.

## Environment

* Python 3.6
* PyTorch 1.1
* CUDA 10

## requirements

lmdb>=0.9
matplotlib>=2.1.0
msgpack==0.5.6
numpy>=1.14.0
pyarrow==0.10.0
tensorflow>=1.12.1
tensorpack==0.8.9

## dataset
"
cd ./dataset
python dataset.py
"
## train the model

python train.py

## test the model

python evaluate.py

## pre-trained model

For baseline PCN*
cd ./pretrained_model/baseline_model

For PCN* + MMI
cd ./pretrained_model/trained_model