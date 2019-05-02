import numpy as np
import os
from utils import create_dir
import tqdm

path_to_pointnet_data = "/mnt/raid/pablo/data/FlyingThings3D/pointcloud_voxelgrid_15_15_10_10_5_25/pointnet_data"
path_to_pointnet_data_features = "/mnt/raid/pablo/data/FlyingThings3D/pointcloud_voxelgrid_15_15_10_10_5_25/pointnet_data_features"

data_splits = ["TEST", "TRAIN"]
letters = ["A", "B", "C"]

SPLIT = "TRAIN"
LETTER = "A"
NUMBER = "0000"
just_one_sample = True


for data_split in data_splits:
    # if data_split != SPLIT and just_one_sample:
    #     continue
    t = tqdm.tqdm(iter(letters), leave=True, total=len(letters), desc='----Letter')
    for _, letter in enumerate(t):
        # if letter != LETTER and just_one_sample:
        #     continue
        for number in sorted(os.listdir(os.path.join(path_to_pointnet_data, data_split, letter))):
            # if number != NUMBER and just_one_sample:
            #     continue
            # if letter == "B" and int(number) < 304:
            #     continue
            path_to_pointnet_data_sequence = os.path.join(path_to_pointnet_data, data_split, letter, number)
            path_to_pointnet_data_features_sequence = os.path.join(path_to_pointnet_data_features, data_split, letter, number)
            create_dir(path_to_pointnet_data_features_sequence)

            for sample in sorted(os.listdir(os.path.join(path_to_pointnet_data_sequence))):
                path_to_pointnet_sample = os.path.join(path_to_pointnet_data_sequence, sample)
                path_to_pointnet_feature_sample = os.path.join(path_to_pointnet_data_features_sequence, sample)

                pointnet_data = np.load(path_to_pointnet_sample)
                # print(path_to_pointnet_feature_sample)
                #
                np.savez(path_to_pointnet_feature_sample,
                         voxels_features=pointnet_data['voxels_features'],
                         voxels_coords=pointnet_data['voxels_coords'],
                         allow_pickle=False)










