import torch.utils.data
import numpy as np
from utils import preprocess_pointcloud, preprocess_voxelnet, preprocess_pointnet
from config import Config as cfg
import os

class PointcloudDataset(torch.utils.data.Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def get_sample(self, pointcloud_t0_path, pointcloud_t1_path, sceneflow_t0_path, sample_name):
        pointcloud_t0_raw = np.load(pointcloud_t0_path)
        pointcloud_t1_raw = np.load(pointcloud_t1_path)
        sceneflow_t0_raw = np.load(sceneflow_t0_path)

        ## Preprocess the pointclouds (compute voxel_features and voxel_coords)
        pcl_data_t0, sf_data_t0 = preprocess_pointcloud(pointcloud_t0_raw['coordinates_np'],
                                                        pointcloud_t0_raw['normal_np'],
                                                        sample_name, sceneflow_t0_raw)
        pcl_data_t1, _ = preprocess_pointcloud(pointcloud_t1_raw['coordinates_np'],
                                               pointcloud_t1_raw['normal_np'],
                                               sample_name)

        return pcl_data_t0, pcl_data_t1, sf_data_t0

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pointcloud_path_t0 = sample[0]
        pointcloud_path_t1 = sample[1]
        sceneflow_path_t0 = sample[2]
        sample_name = sample[3].replace('\\', '/')
        sample_name = sample_name.replace('/', '-')

        ###########################
        ## pcl_data := pointcloud
        ## sf  := sceneflow
        ###########################
        pcl_data_t0, pcl_data_t1, sf_gt = self.get_sample(pointcloud_path_t0, pointcloud_path_t1,
                                                          sceneflow_path_t0, sample_name)

        return pcl_data_t0, pcl_data_t1, sf_gt, sample_name


class SiameseBaselineDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def get_sample(self, pcl_data_path_t0, pcl_data_path_t1, sf_data_path_gt):
        pcl_data_t0 = np.load(pcl_data_path_t0)
        pcl_data_t1 = np.load(pcl_data_path_t1)
        sf_data_t0 = np.load(sf_data_path_gt)

        return pcl_data_t0['voxel_coords'], \
               pcl_data_t1['voxel_coords'], \
               sf_data_t0['voxel_sceneflows']

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pcl_data_path_t0 = sample[0]
        pcl_data_path_t1 = sample[1]
        sceneflow_path_t0 = sample[2]
        sample_name = sample[3].replace('\\', '/')
        sample_name = sample_name.replace('/', '-')

        pcl_data_t0, pcl_data_t1, sf_data_gt = self.get_sample(pcl_data_path_t0, pcl_data_path_t1,
                                                               sceneflow_path_t0)
        return pcl_data_t0, pcl_data_t1, sf_data_gt, sample_name


class SiameseBaselineDatasetTest(torch.utils.data.Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def get_sample(self, pcl_data_path_t0, pcl_data_path_t1, sf_data_path_gt):
        pcl_data_t0 = np.load(pcl_data_path_t0)
        pcl_data_t1 = np.load(pcl_data_path_t1)
        sf_data_t0 = np.load(sf_data_path_gt)

        return (pcl_data_t0['points'], pcl_data_t0['voxel_coords'],
                pcl_data_t0['inv_ind']), \
               (pcl_data_t1['points'], pcl_data_t1['voxel_coords'],
                None), \
               (sf_data_t0['sceneflow_np'], sf_data_t0['voxel_sceneflows'])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pcl_data_path_t0 = sample[0]
        pcl_data_path_t1 = sample[1]
        sceneflow_path_t0 = sample[2]
        sample_name = sample[3].replace('\\', '/')
        sample_name = sample_name.replace('/', '-')

        pcl_data_t0, pcl_data_t1, sf_data_gt = self.get_sample(pcl_data_path_t0, pcl_data_path_t1,
                                                               sceneflow_path_t0)
        return pcl_data_t0, pcl_data_t1, sf_data_gt, sample_name


class SiamesePointNetDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def get_sample(self, pcl_data_path_t0, pcl_data_path_t1, sf_data_path_gt):
        pcl_data_t0 = np.load(pcl_data_path_t0)
        pcl_data_t1 = np.load(pcl_data_path_t1)
        sf_data_t0 = np.load(sf_data_path_gt)

        return (pcl_data_t0['voxels_features'], pcl_data_t0['voxels_coords']),\
               (pcl_data_t1['voxels_features'], pcl_data_t1['voxels_coords']),\
               sf_data_t0['voxel_sceneflows']

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pcl_data_path_t0 = sample[0]
        pcl_data_path_t1 = sample[1]
        sceneflow_path_t0 = sample[2]
        sample_name = sample[3].replace('\\', '/')
        sample_name = sample_name.replace('/', '-')

        pcl_data_t0, pcl_data_t1, sf_data_gt = self.get_sample(pcl_data_path_t0, pcl_data_path_t1,
                                                               sceneflow_path_t0)
        return pcl_data_t0, pcl_data_t1, sf_data_gt, sample_name


class SiamesePointNetDatasetTest(torch.utils.data.Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def get_base_path(self, path_to_data, type):
        pcl_data_base = os.path.split(path_to_data)
        sample = []
        sample.append(pcl_data_base[1])

        for i in range(4):
            pcl_data_base = os.path.split(pcl_data_base[0])
            if i < 3:
                sample = [pcl_data_base[1]] + sample

        if type == 'pcl':
            pcl_data_base = os.path.join(pcl_data_base[0], 'pointcloud_voxelgrid')
        elif type == 'sf':
            pcl_data_base = os.path.join(pcl_data_base[0], 'sceneflow')

        for i in range(4):
            pcl_data_base = os.path.join(pcl_data_base, sample[i])
        return pcl_data_base

    def get_sample(self, pcl_data_path_t0, pcl_data_path_t1, sf_data_path_gt):
        pcl_raw_path_t0 = self.get_base_path(pcl_data_path_t0, 'pcl')
        pcl_raw_path_t1 = self.get_base_path(pcl_data_path_t1, 'pcl')
        sf_raw_path_gt = self.get_base_path(sf_data_path_gt, 'sf')

        pcl_raw_t0 = np.load(pcl_raw_path_t0)
        pcl_raw_t1 = np.load(pcl_raw_path_t1)
        sf_raw_t0 = np.load(sf_raw_path_gt)
        pcl_data_t0 = np.load(pcl_data_path_t0)
        pcl_data_t1 = np.load(pcl_data_path_t1)
        sf_data_t0 = np.load(sf_data_path_gt)

        return (pcl_raw_t0['points'], pcl_data_t0['voxels_features'],
                pcl_data_t0['voxels_coords'], pcl_raw_t0['inv_ind']),\
               (pcl_raw_t1['points'], pcl_data_t1['voxels_features'],
                pcl_data_t1['voxels_coords'], pcl_raw_t1['inv_ind']),\
               (sf_raw_t0['sceneflow_np'], sf_data_t0['voxel_sceneflows'])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pcl_data_path_t0 = sample[0]
        pcl_data_path_t1 = sample[1]
        sceneflow_path_t0 = sample[2]
        sample_name = sample[3].replace('\\', '/')
        sample_name = sample_name.replace('/', '-')

        #####################
        ## vg := voxelgrid ##
        ## sf := sceneflow ##
        #####################
        pcl_data_t0, pcl_data_t1, sf_data_gt = self.get_sample(pcl_data_path_t0, pcl_data_path_t1,
                                                               sceneflow_path_t0)
        return pcl_data_t0, pcl_data_t1, sf_data_gt, sample_name