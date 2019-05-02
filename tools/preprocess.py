import numpy as np
#np.set_printoptions(threshold=np.nan)
import os
from models_pointnet import QueryAndGroup, PointNetMLP
from config import Config as cfg
import torch.utils.data
from utils import create_dir

torch.set_printoptions(precision=8)



path_to_pcl_data = "/mnt/raid/pablo/data/FlyingThings3D/pointcloud_voxelgrid_15_15_10_10_5_40/pointcloud_voxelgrid"

path_to_voxels_xyz_features = "/mnt/raid/pablo/data/FlyingThings3D/pointcloud_voxelgrid_15_15_10_10_5_40/voxels_xyz_features"
path_to_voxels_xyz_normals_features = "/mnt/raid/pablo/data/FlyingThings3D/pointcloud_voxelgrid_15_15_10_10_5_40/voxels_xyz_normals_features"
path_to_voxels_features = "/mnt/raid/pablo/data/FlyingThings3D/pointcloud_voxelgrid_15_15_10_10_5_40/voxels_features"
path_to_voxels_features_normals = "/mnt/raid/pablo/data/FlyingThings3D/pointcloud_voxelgrid_15_15_10_10_5_40/voxels_features_normals"

# data_splits = ["TEST", "TRAIN"]
# letters = ["A", "B", "C"]

data_splits = ["TRAIN"]
letters = ["B"]

nsample = 16
model = QueryAndGroup(npoints=cfg.T, radius=0.4, nsample=nsample)
mlp = PointNetMLP(mlp_spec=[6, cfg.nfeat // 2, cfg.nfeat // 2, cfg.nfeat])


SPLIT = "TRAIN"
LETTER = "B"
NUMBER = "0594"
just_one_sample = True

print("hallo")


with torch.no_grad():
    model.eval()
    model.cuda()

    # mlp.eval()
    # mlp.cuda()

    for data_split in data_splits:
        if data_split != SPLIT and just_one_sample:
            continue
        for letter in letters:
            if letter != LETTER and just_one_sample:
                continue
            for number in sorted(os.listdir(os.path.join(path_to_pcl_data, data_split, letter))):
                if number != NUMBER and just_one_sample:
                    continue

                if int(number) < 585 or int(number) > 599:
                    continue

                path_to_pcl_sequence = os.path.join(path_to_pcl_data, data_split, letter, number)
                print(path_to_pcl_sequence)
                path_to_voxels_xyz_features_sequence = os.path.join(path_to_voxels_xyz_features, data_split, letter, number)
                path_to_voxels_xyz_normals_features_sequence = os.path.join(path_to_voxels_xyz_normals_features, data_split, letter, number)
                path_to_voxels_features_sequence = os.path.join(path_to_voxels_features, data_split, letter, number)
                path_to_voxels_features_normals_sequence = os.path.join(path_to_voxels_features_normals, data_split, letter, number)

                create_dir(path_to_voxels_xyz_features_sequence)
                create_dir(path_to_voxels_xyz_normals_features_sequence)
                create_dir(path_to_voxels_features_sequence)
                create_dir(path_to_voxels_features_normals_sequence)

                ######################################################
                ######################################################
                ###################################################
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()

                for sample in sorted(os.listdir(os.path.join(path_to_pcl_sequence))):
                    path_to_pcl_sample = os.path.join(path_to_pcl_sequence, sample)

                    path_to_voxels_xyz_features_sample = os.path.join(path_to_voxels_xyz_features_sequence, sample)
                    path_to_voxels_xyz_normals_features_sample = os.path.join(path_to_voxels_xyz_normals_features_sequence, sample)
                    path_to_voxels_features_sample = os.path.join(path_to_voxels_features_sequence, sample)
                    path_to_voxels_features_normals_sample = os.path.join(path_to_voxels_features_normals_sequence, sample)

                    #print()
                    # print(path_to_voxels_xyz_features_sample)
                    # print(path_to_voxels_xyz_normals_features_sample)
                    # print(path_to_voxels_features_sample)
                    # print(path_to_voxels_features_normals_sample)

                    pcl_data = np.load(path_to_pcl_sample)

                    points = pcl_data['points']
                    normals = pcl_data['normals']
                    voxel_coords = pcl_data['voxel_coords']
                    inv_ind = pcl_data['inv_ind']

                    voxels_xyz_features = []
                    voxels_xyz_normals_features = []
                    voxels_features = []
                    voxels_features_normals = []

                    points_cuda = torch.cuda.FloatTensor(points).unsqueeze(0)
                    normals_cuda = torch.cuda.FloatTensor(normals).unsqueeze(0)

                    for i in range(len(voxel_coords)):
                        mask = inv_ind == i

                        pts = points[mask]
                        num_pts = pts.shape[0]
                        pts = torch.cuda.FloatTensor(pts)
                        pts = pts.unsqueeze(0)

                        n = normals[mask]
                        n = torch.cuda.FloatTensor(n)
                        n = n.unsqueeze(0)

                        ## Compute centroids and features
                        new_xyz, new_normals, new_features, new_features_normals = \
                            model((points_cuda, normals_cuda, pts, n))

                        print(new_xyz.shape)
                        print(new_normals.shape)
                        print(new_features.shape)
                        print(new_features_normals.shape)

                        # new_xyz w/o normals
                        new_xyz_centered = new_xyz - new_xyz.mean(dim=1)
                        new_xyz_concat = torch.cat((new_xyz, new_xyz_centered), 2)
                        new_xyz_feature = new_xyz_concat.unsqueeze(3).permute(0, 2, 1, 3)

                        # new_xyz w/ normals
                        new_xyz_normals_concat = torch.cat((new_xyz, new_xyz_centered, new_normals), 2)
                        new_xyz_normals_feature = new_xyz_normals_concat.unsqueeze(3).permute(0, 2, 1, 3)


                        print(new_xyz_feature.shape, new_xyz_normals_feature.shape)

                        new_xyz_feature = new_xyz_feature.permute(0, 2, 3, 1)
                        new_xyz_normals_feature = new_xyz_normals_feature.permute(0, 2, 3, 1)
                        new_features = new_features.permute(0, 2, 3, 1)
                        new_features_normals = new_features_normals.permute(0, 2, 3, 1)

                        #print(new_xyz_feature.shape, new_xyz_normals_feature.shape, new_features.shape, new_features_normals.shape)

                        ## Create mask
                        mask_repeated = np.zeros(cfg.T)
                        mask_repeated[:num_pts] = 1
                        mask_repeated = torch.cuda.FloatTensor(mask_repeated)

                        mask_new_xyz_feature = mask_repeated.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, 6)
                        mask_new_xyz_normals_feature = mask_repeated.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, 9)
                        mask_new_features = mask_repeated.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1, 1, nsample, 3)
                        mask_new_features_normals = mask_repeated.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1, 1, nsample, 6)

                        # print(mask_new_xyz_feature.shape, mask_new_xyz_normals_feature.shape, mask_new_features.shape, mask_new_features_normals.shape)
                        # print(mask_new_xyz_feature, mask_new_xyz_normals_feature, mask_new_features, mask_new_features_normals)


                        ## Mask points and features

                        new_xyz_feature *= mask_new_xyz_feature
                        new_xyz_normals_feature *= mask_new_xyz_normals_feature
                        new_features *= mask_new_features
                        new_features_normals *= mask_new_features_normals

                        new_xyz_feature = new_xyz_feature.permute(0, 3, 1, 2)
                        new_xyz_normals_feature = new_xyz_normals_feature.permute(0, 3, 1, 2)
                        new_features = new_features.permute(0, 3, 1, 2)
                        new_features_normals = new_features_normals.permute(0, 3, 1, 2)

                        print(new_xyz_feature.shape, new_xyz_normals_feature.shape, new_features.shape, new_features_normals.shape)

                        voxels_xyz_features.append(new_xyz_feature.cpu().numpy())
                        voxels_xyz_normals_features.append(new_xyz_normals_feature.cpu().numpy())
                        voxels_features.append(new_features.cpu().numpy())
                        voxels_features_normals.append(new_features_normals.cpu().numpy())



                    voxels_xyz_features = np.concatenate(voxels_xyz_features)
                    voxels_xyz_normals_features = np.concatenate(voxels_xyz_normals_features)
                    voxels_features = np.concatenate(voxels_features)
                    voxels_features_normals = np.concatenate(voxels_features_normals)

                    ##################################################
                    ##################################################

                    # voxels_features_normals = torch.cuda.FloatTensor(voxels_features_normals)
                    # print(voxels_features_normals.shape)
                    # print(voxels_features_normals[0])
                    #
                    # # voxels_features_normals = voxels_features_normals.unsqueeze(0)
                    # # print(voxels_features_normals.shape)
                    #
                    # voxels_features_normals = mlp(voxels_features_normals)
                    # print(voxels_features_normals.shape)
                    # print(voxels_features_normals[0])

                    ##################################################
                    #####################################################

                    # np.savez(path_to_voxels_xyz_features_sample,
                    #          voxels_features=voxels_xyz_features,
                    #          voxels_coords=voxel_coords,
                    #          allow_pickle=False)
                    #
                    # np.savez(path_to_voxels_xyz_normals_features_sample,
                    #          voxels_features=voxels_xyz_normals_features,
                    #          voxels_coords=voxel_coords,
                    #          allow_pickle=False)
                    #
                    # np.savez(path_to_voxels_features_sample,
                    #          voxels_features=voxels_features,
                    #          voxels_coords=voxel_coords,
                    #          allow_pickle=False)
                    #
                    # np.savez(path_to_voxels_features_normals_sample,
                    #          voxels_features=voxels_features_normals,
                    #          voxels_coords=voxel_coords,
                    #          allow_pickle=False)

                ###################################################
                # end.record()
                # torch.cuda.synchronize()
                # print("timing voxelNet: ", start.elapsed_time(end))
                # ###################################################










