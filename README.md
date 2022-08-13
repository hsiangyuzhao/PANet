# Prior Attention Network for Multi-Lesion Segmentation in Medical Images
### :tada: Our work has been accepted by *IEEE Transactions on Medical Imaging*  
**Authors:**  
> Xiangyu Zhao[1][2][3], Peng Zhang[1][2], Fan Song[1][2], Chenbin Ma[1][2], Guangda Fan[1][2], Yangyang Sun[1][2], Youdan Feng[1][2], Guanglei Zhang[1][2]*  

**Institution:**
> [1] School of Biological Science and Medical Engineering, Beihang University, Beijing, China  
> [2] Beijing Advanced Innovation Center for Biomedical Engineering, Beihang University, Beijing, China  
> [3] School of Biomedical Engineering, Shanghai Jiao Tong University, Shanghai, China  
> *Corresponding Author: Guanglei Zhang

manuscript link: https://arxiv.org/abs/2110.04735  
This repo contains the implementation of 3D segmentation of BraTS 2020 with the proposed *Prior Attention Network*.  
**If you use our code, please cite the paper:**  
> @ARTICLE{9852260,  
  author={Zhao, Xiangyu and Zhang, Peng and Song, Fan and Ma, Chenbin and Fan, Guangda and Sun, Yangyang and Feng, Youdan and Zhang, Guanglei},  
  journal={IEEE Transactions on Medical Imaging},   
  title={Prior Attention Network for Multi-Lesion Segmentation in Medical Images},   
  year={2022},  
  volume={},  
  number={},  
  pages={1-1},  
  doi={10.1109/TMI.2022.3197180}}  

## Methods
In this paper we propose a novel *Prior Attention Network* with intermediate supervision, parameterized skip connections and deep supervision strategy to address multi-lesion segmentation problems in medical images.  
### Network Topology
![network](https://user-images.githubusercontent.com/53631393/136913718-e94f7ba1-8444-4445-8682-692ff6a99a62.png)
### Attention Guiding Decoder
![AGD](https://user-images.githubusercontent.com/53631393/136913725-04e109d3-8081-49ca-948c-54e866692200.png)

## Results
### Quantitative Results
![Snipaste_2021-10-12_15-47-15](https://user-images.githubusercontent.com/53631393/136914282-3dd5a697-711b-4653-adb8-a6d2c98705f5.png)
### Qualitative Results
![vis3d](https://user-images.githubusercontent.com/53631393/136914543-023500b6-9a57-4f21-9f94-77961c7e9917.png)
### Ablation Analysis
![Snipaste_2021-10-12_15-47-32](https://user-images.githubusercontent.com/53631393/136914298-b76690c2-987d-4a3b-98da-9ab42f44ed10.png)

## Usage
### Data Preparation
Please download BraTS 2020 data according to `https://www.med.upenn.edu/cbica/brats2020/data.html`.  
Unzip downloaded data at `./data` folder (please create one) and remove all the csv files in the folder, or it will cause errors.

### Pretrained Checkpoint
We provide ckpt download via Google Drive or Baidu Netdisk. Please download the checkpoint from the url below:  
#### Google Drive
url: https://drive.google.com/file/d/16Gy5mMzMPLvt1jVgzBsmBUbZTrUCKtWv/view?usp=sharing
#### Baidu Netdisk
url：https://pan.baidu.com/s/14qM2k46mFnzT2RmI3sWcSw  
extraction code：0512  

### Training
For default training configuration, we use patch-based training pipeline and use Adam optimizer. Deep supervision is utilized to facilitate convergence.
#### Train and validate on BraTS training set
```python
python train.py --model panet --patch_test --ds
```
#### Training on the entire BraTS training set
```python
python train.py --model panet --patch_test --ds --trainset
```
#### Breakpoint continuation for training
```python
python train.py --model panet --patch_test --ds -c CKPT
```
this will load the pretrained weights as well as the status of optimizer and scheduler.
#### PyTorch-native AMP training
```python
python train.py --model panet --patch_test --ds --mixed
```
if the training is too slow, please enable CUDNN benchmark by adding `--benchmark` but it will slightly affects the reproducibility.

### Inference
For default inference configuration, we use patch-based pipeline.
```python
python inference.py --model panet --patch_test --validation -c CKPT
```
