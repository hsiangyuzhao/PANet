# Prior Attention Network for Multi-Lesion Segmentation in Medical Images
**Authors: Xiangyu Zhao, Peng Zhang, Fan Song, Chenbin Ma, Guangda Fan, Yangyang Sun, Youdan Feng, Guanglei Zhang**  
**Institution: School of Biological Science and Medical Engineering, Beihang University; Beijing Advanced Innovation Center for Biomedical Engineering, Beihang University**  
https://arxiv.org/abs/2110.04735  
This repo contains the implementation of 3D segmentation of BraTS 2020 with the proposed *Prior Attention Network*.

## Methods
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
Please download the checkpoint from the url below:  
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
#### PyTorch-native AMP training
```python
python train.py --model panet --patch_test --ds --mixed
```

### Inference
For default inference configuration, we use patch-based pipeline.
```python
python inference.py --model panet --patch_test --validation -c CKPT
```
