# How to use Flow for Motion Segmentation, Instance Segmentation and Pose Estimation


# Installation
- Install [Fast Geodis](https://github.com/masadcv/FastGeodis) with pip install FastGeodis --no-build-isolation
- Install [PyTorch3d](https://github.com/facebookresearch/pytorch3d) with CUDA support.
- Install [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter/tree/master) with CUDA support.

# Benchmark data
- Setup directory for extracting the data, visuals and experimental results
'''console
BASE_PATH='path_where_to_store_data'
'''
- Download [Data](https://login.rci.cvut.cz/data/lidar_intensity/sceneflow/data_sceneflow.tgz) and unpack it to the folder $BASE_PATH/

```console
tar -xvf data_sceneflow.tgz $BASE_PATH/data/sceneflow
```

The data consist of *.npz files, where inside key names corresponds to:
'pc1' : xyz points in time **t**
'pc2' : xyz points in time **t+1**
'pose1' : Pose Transformation from **t** to **t+1** 

# Use Case

After installation of the package and setting up the data, you can run: 
```console
python optimize_frame.py *path_to_frame*
```

The script will compute Flows, dynamic mask per-point, instances by DBSCAN from geometry and motion features and pose estimation in output:

```console
Eval time:  3.917 
Clusters:  (136898,) ---> [int ids]
Dynamic points:  torch.Size([136898]) ---> [binary mask]
Flow:  torch.Size([136898, 3]) ---> [float per-point flows]
Pose:  torch.Size([4, 4]) ---> [odometry matrix]
```

where you can look at the and of the script for format and save the output to your desired location.