from config import Config
import platform
import numpy as np
from PlyFile import PlyFile
import copy
import os

system = platform.system()
cfg = Config(system=system)

pointcloud_t0_path = cfg.data_dir['pointcloud'] + "/pointcloud/TRAIN/A/0000/0006.npy"
pointcloud_t1_path = cfg.data_dir['pointcloud'] + "/pointcloud/TRAIN/A/0000/0007.npy"
sceneflow_t0_path = cfg.data_dir["pointcloud"] + "/sceneflow/TRAIN/A/0000/0006.npy"

print(pointcloud_t0_path, pointcloud_t1_path, sceneflow_t0_path)

pcl_t0 = np.load(pointcloud_t0_path)
pcl_t1 = np.load(pointcloud_t1_path)
sf_t0 = np.load(sceneflow_t0_path)

flowed_points = pcl_t0 + sf_t0


print(pcl_t0.shape)
print(pcl_t1.shape)
print(sf_t0.shape)

logdir = os.path.join(cfg.log_dir_base["pointcloud"])

plyfile = PlyFile(pcl_t0, color=[255, 255, 255])
plyfile.append_points(pcl_t1, color=[255, 0, 0])
plyfile.append_points(flowed_points, color=[0, 0, 0])
plyfile.write_ply(logdir + "/quick_tests/align_pcl_only_pcl.ply")

######

# vg_t0_path = cfg.data_vg_dir + "/voxelgrid/TRAIN/A/0000/0006.npy"
# vg_t1_path = cfg.data_vg_dir + "/voxelgrid/TRAIN/A/0000/0007.npy"
# sf_t0_path = cfg.data_vg_dir + "/sceneflow/TRAIN/A/0000/0006.npy"
#
# vg_t0 = np.load(vg_t0_path)
# vg_t0_raw = copy.deepcopy(vg_t0)
# vg_t0 = np.argwhere(vg_t0)
#
# vg_t1 = np.load(vg_t1_path)
# vg_t1 = np.argwhere(vg_t1)
#
# sf_t0 = np.load(sf_t0_path)
# sf_t0 = sf_t0[np.where(vg_t0_raw == 1)]
#
# flowed_voxels = vg_t0 + sf_t0
#
# print(vg_t0.shape)
# print(vg_t1.shape)
# print(sf_t0.shape)
#
# plyfile = PlyFile(vg_t0, color=[255, 0 , 0])
# plyfile.append_points(vg_t1, color=[0, 255, 0])
# plyfile.append_points(flowed_voxels, color=[0, 0, 255])
# plyfile.write_ply("/mnt/raid/pablo/experiments/quick_tests/align_vg.ply")