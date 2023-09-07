# Pseudo-LiDAR-for-Visual-Odometry
TIM2023 "Pseudo-LiDAR for Visual Odometry" created by Yanzi Miao, Huiying Deng, Chaokang Jiang, Zhiheng Feng, Xinrui Wu, Guangming Wang, and Hesheng Wang.
<img src="pipeline2.png">

## Installation
Our model only depends on the following commonly used packages.

| Package      | Version                          |
| ------------ | -------------------------------- |
| CUDA         |  11.3                            |
| PyTorch      |  1.10.0                          |
| h5py         | *not specified*                  |
| tqdm         | *not specified*                  |
| numpy        | *not specified*                  |
| openpyxl      | *not specified*                  |

Device: NVIDIA RTX 3090

## Install the pointnet2 library
Compile the furthest point sampling, grouping and gathering operation for PyTorch with following commands. 
```bash
cd ops_pytorch
cd fused_conv_random_k
python setup.py install
cd ../
cd fused_conv_select_k
python setup.py install
cd ../
```
## Datasets
### KITTI Dataset
Datasets are available at KITTI Odometry benchmark website: [ https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
The data of the KITTI odometry dataset should be organized as follows: 

```
data_root
├── 00
│   ├── velodyne
│   ├── calib.txt
├── 01
├── ...
```


## Training
Train the network by running :
```bash
python train.py 
```
Please reminder to specify the `GPU`, `data_root`,`log_dir`, `train_list`(sequences for training), `val_list`(sequences for validation).
You may specify the value of arguments. Please find the available arguments in the configs.py. 

## Testing
Our network is evaluated every 5 epoph during training. If you only want the evaluation results, you can set the parameter 'eval_before' as 'True' in file config.py, then evaluate the network by running :
```bash
python train.py
```
Please reminder to specify the `GPU`, `data_root`,`log_dir`, `test_list`(sequences for testing) in the scripts.
You can also get the pretrined model in https://drive.google.com/file/d/13xQGTcsO0YwBYbcQTSHk7BkNrOLG18mX/view?usp=drive_link.

## Quantitative results:
### KITTI 
<img src="kitti.png">


## Citation
```
@article{deng2022pseudo,
  title={Pseudo-LiDAR for Visual Odometry},
  author={Miao, Yanzi and Deng, Huiying and Jiang, Chaokang and Feng, Zhiheng and  and Wu, Xinrui and Wang, Guangming and Wang, Hesheng},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2022}
}
```
### Acknowledgments
We thank the following open-source project for the help of the implementations:
- [PointNet++](https://github.com/charlesq34/pointnet2) 
- [KITTI_odometry_evaluation_tool](https://github.com/LeoQLi/KITTI_odometry_evaluation_tool) 
- [PWCLONet] (https://github.com/IRMVLab/PWCLONet)

