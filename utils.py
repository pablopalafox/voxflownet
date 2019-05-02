import re
from scipy import misc
import numpy as np
# np.set_printoptions(threshold=np.nan)
import sys
import pandas as pd
import os
from config import Config as cfg
from libs.pyntcloud.pyntcloud import PyntCloud
import glob
from sklearn.model_selection import train_test_split
from itertools import compress

from config import DATA_TYPES_3D
from sklearn.decomposition import PCA
from colorama import Fore, Back, Style


######################################################################################################
######################################################################################################


def read(file):
    if file.endswith('.float3'): return readFloat(file)
    elif file.endswith('.flo'): return readFlow(file)
    elif file.endswith('.ppm'): return readImage(file)
    elif file.endswith('.pgm'): return readImage(file)
    elif file.endswith('.png'): return readImage(file)
    elif file.endswith('.jpg'): return readImage(file)
    elif file.endswith('.pfm'): return readPFM(file)[0]
    else: raise Exception('don\'t know how to read %s' % file)


def write(file, data):
    if file.endswith('.float3'): return writeFloat(file, data)
    elif file.endswith('.flo'): return writeFlow(file, data)
    elif file.endswith('.ppm'): return writeImage(file, data)
    elif file.endswith('.pgm'): return writeImage(file, data)
    elif file.endswith('.png'): return writeImage(file, data)
    elif file.endswith('.jpg'): return writeImage(file, data)
    elif file.endswith('.pfm'): return writePFM(file, data)
    else: raise Exception('don\'t know how to write %s' % file)


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)


def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)[0]
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data

    return misc.imread(name)


def writeImage(name, data):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return writePFM(name, data, 1)

    return misc.imsave(name, data)


def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


def readFloat(name):
    f = open(name, 'rb')

    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data


def writeFloat(name, data):
    f = open(name, 'wb')

    dim=len(data.shape)
    if dim>3:
        raise Exception('bad float file dimension: %d' % dim)

    f.write(('float\n').encode('ascii'))
    f.write(('%d\n' % dim).encode('ascii'))

    if dim == 1:
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
    else:
        f.write(('%d\n' % data.shape[1]).encode('ascii'))
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
        for i in range(2, dim):
            f.write(('%d\n' % data.shape[i]).encode('ascii'))

    data = data.astype(np.float32)
    if dim==2:
        data.tofile(f)

    else:
        np.transpose(data, (2, 0, 1)).tofile(f)


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################


def getBlackListDirs():
    black_dirs_txt = "black_list_dirs.txt"
    with open(black_dirs_txt, 'r') as f:
        black_dirs = f.read().splitlines()
    return black_dirs


def load_sequence(datatype3d_base_dir, sceneflow_base_dir, sample_base_dir, data_type):
    samples = []

    sceneflow_dir = os.path.join(sceneflow_base_dir, sample_base_dir)
    for path in sorted(glob.glob(sceneflow_dir + "/*")):

        sceneflow_path = path.replace('\\', '/')

        sample_number_0 = os.path.basename(path).split('.')[0]
        if data_type == DATA_TYPES_3D['POINTCLOUD']:
            sample_path_0 = os.path.join(sample_base_dir, sample_number_0 + ".npy")
        elif data_type == DATA_TYPES_3D['BOTH']:
            sample_path_0 = os.path.join(sample_base_dir, sample_number_0 + ".npz")

        sample_name = sample_base_dir.replace('/', '-') + "-" + sample_number_0

        sample_number_1 = str(int(os.path.basename(path).split('.')[0]) + 1).zfill(4)
        if data_type == DATA_TYPES_3D['POINTCLOUD']:
            sample_path_1 = os.path.join(sample_base_dir, sample_number_1 + ".npy")
        elif data_type == DATA_TYPES_3D['BOTH']:
            sample_path_1 = os.path.join(sample_base_dir, sample_number_1 + ".npz")

        datatype3d_path_0 = os.path.join(datatype3d_base_dir, sample_path_0)
        datatype3d_path_1 = os.path.join(datatype3d_base_dir, sample_path_1)

        sample = [datatype3d_path_0, datatype3d_path_1, sceneflow_path, sample_name]
        samples.append(sample)

    return samples


def sequence_exists(sceneflow_base_dir, sample_base_dir):
    """
    Returns whether or not the path to a sequence exists
    :param sceneflow_base_dir:
    :param sample_base_dir:
    :return:
    """
    sequence_path = os.path.join(sceneflow_base_dir, sample_base_dir)
    if os.path.isdir(sequence_path):
        return True
    else:
        return False


def check_sequence_number(number):
    """
    Checks if the sequence number ''number'' is a valid one
    :param number:
    :return:
    """
    if number >= 750:
        raise Exception("Sequences range from 0000 to 0749")


def load_files(input_base_dir, sceneflow_base_dir, data_split, data_type, sequences_to_use):
    """
    Load numpy files containing the voxelgrids and the sceneflow groundtruth
    :param dataset_path:
    :return: list of path files for the voxelgrids and the sceneflow groungtruth
    """

    black_list_dirs = getBlackListDirs()

    all_samples = []

    if sequences_to_use == "ALL":
        ## Use the whole dataset
        for letter in os.listdir(os.path.join(sceneflow_base_dir, data_split)):
            for number in os.listdir(os.path.join(sceneflow_base_dir, data_split, letter)):
                sequence = os.path.join(letter, number)
                sample_base_dir = os.path.join(data_split, sequence).replace('\\', '/')
                if sample_base_dir in black_list_dirs:
                    continue
                sequence_samples = load_sequence(input_base_dir, sceneflow_base_dir, sample_base_dir, data_type)
                all_samples.append(sequence_samples)
    else:
        for sequence_to_use in sequences_to_use:
            if sequence_to_use == "A" or sequence_to_use == "B" or sequence_to_use == "C":
                """Get a complete letter"""
                letter = sequence_to_use
                for number in os.listdir(os.path.join(sceneflow_base_dir, data_split, letter)):
                    sequence = os.path.join(letter, number)
                    sample_base_dir = os.path.join(data_split, sequence).replace('\\', '/')
                    if sample_base_dir in black_list_dirs:
                        continue
                    sequence_samples = load_sequence(input_base_dir, sceneflow_base_dir, sample_base_dir, data_type)
                    all_samples.append(sequence_samples)
            elif "-" in sequence_to_use:
                letter, numbers_range = sequence_to_use.split('/')
                _from, _to = numbers_range.split('-')
                _from, _to = int(_from), int(_to)
                check_sequence_number(_from)
                check_sequence_number(_to)
                for number in range(_from, _to + 1):
                    number = str(number).zfill(4)
                    sequence = os.path.join(letter, number)
                    sample_base_dir = os.path.join(data_split, sequence).replace('\\', '/')
                    if sample_base_dir in black_list_dirs or not sequence_exists(sceneflow_base_dir, sample_base_dir):
                        continue
                    sequence_samples = load_sequence(input_base_dir, sceneflow_base_dir, sample_base_dir, data_type)
                    all_samples.append(sequence_samples)
            else:
                number = int(sequence_to_use.split('/')[1])
                check_sequence_number(number)
                sample_base_dir = os.path.join(data_split, sequence_to_use).replace('\\', '/')
                if sample_base_dir in black_list_dirs:
                    raise Exception("Sequence to eval is in Black List!")
                sequence_samples = load_sequence(input_base_dir, sceneflow_base_dir, sample_base_dir, data_type)
                all_samples.append(sequence_samples)

    final_samples = []
    for sequence_samples in all_samples:
        for sample in sequence_samples:
            final_samples.append(sample)

    return final_samples


def get_train_val_loader(dataset_dir, data_split, data_type, use_local, use_normal,
                         sequences_to_train=None, batch_size_train=1, batch_size_val=1,
                         validation_percentage=0.05):
    """
    Compute dataset loader
    :param dataset_dir:
    :param batch_size:
    :return:
    """
    import torch.utils.data
    from torch.utils.data.dataloader import default_collate

    if cfg.model_name == "SiameseModel3D":
        detection_collate = detection_collate_baseline_train
    elif cfg.model_name == "SiamesePointNet":
        detection_collate = detection_collate_pointnet_train

    if data_type == DATA_TYPES_3D['POINTCLOUD']:
        from loader import PointcloudDataset as Dataset
    elif data_type == DATA_TYPES_3D['BOTH']:
        if cfg.model_name == "SiameseModel3D":
            from loader import SiameseBaselineDatasetTrain as Dataset
        elif cfg.model_name == "SiamesePointNet":
            from loader import SiamesePointNetDatasetTrain as Dataset


    ## Load files lists
    if cfg.model_name == "SiameseModel3D":
        vg_or_pcl_dir = os.path.join(dataset_dir, "pointcloud_voxelgrid")
    else:
        if use_local:
            if use_normal:
                vg_or_pcl_dir = os.path.join(dataset_dir, "voxels_features_normals")
            else:
                vg_or_pcl_dir = os.path.join(dataset_dir, "voxels_features")
        else:
            if use_normal:
                vg_or_pcl_dir = os.path.join(dataset_dir, "voxels_xyz_normals_features")
            else:
                vg_or_pcl_dir = os.path.join(dataset_dir, "voxels_xyz_features")

    sceneflow_dir = os.path.join(dataset_dir, "sceneflow")

    samples = load_files(vg_or_pcl_dir, sceneflow_dir, data_split, data_type, sequences_to_train)
    samples_train, samples_val = train_test_split(samples, test_size=validation_percentage,
                                                  random_state=20)

    #####################################################################
    ## HELP: DO NOT REMOVE - USE TO GET THE SAMPLES IN VALIDATION SET ###
    #####################################################################
    # validation_samples = []
    # for sample_val in samples_val:
    #     validation_samples.append(sample_val[-1])
    # validation_samples.sort()
    # with open("validation_samples.txt", "w") as f:
    #     for sample in validation_samples:
    #         f.write(sample + "\n")
    #####################################################################
    #####################################################################

    ## Create TRAIN loader
    train_dataset = Dataset(samples_train)
    print("Train Dataset's length:", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                                               num_workers=8, collate_fn=detection_collate,
                                               drop_last=True, pin_memory=False)

    ## Create VAL loader
    val_dataset = Dataset(samples_val)
    print("Val Dataset's length:", len(val_dataset))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True,
                                             num_workers=8, collate_fn=detection_collate,
                                             drop_last=True, pin_memory=False)

    print("Number of training batches: ", len(train_loader),
          "(Samples: ", str(len(train_loader) * batch_size_train), ")")
    print("Number of val batches: ", len(val_loader),
          "(Samples: ", str(len(val_loader) * batch_size_val), ")")

    return train_loader, val_loader


def get_eval_loader(dataset_dir, data_split, data_type, use_local, use_normal,
                    sequences_to_eval=None, batch_size=1):
    """
    Compute dataset loader
    :param dataset_dir:
    :param batch_size:
    :return:
    """
    import torch.utils.data
    from torch.utils.data.dataloader import default_collate

    if cfg.model_name == "SiameseModel3D":
        detection_collate = detection_collate_baseline_test
    elif cfg.model_name == "SiamesePointNet":
        detection_collate = detection_collate_pointnet_test

    if data_type == DATA_TYPES_3D['POINTCLOUD']:
        from loader import PointcloudDataset as Dataset
    elif data_type == DATA_TYPES_3D['BOTH']:
        if cfg.model_name == "SiameseModel3D":
            from loader import SiameseBaselineDatasetTest as Dataset
        elif cfg.model_name == "SiamesePointNet":
            from loader import SiamesePointNetDatasetTest as Dataset

    ## Load files lists
    if cfg.model_name == "SiameseModel3D":
        vg_or_pcl_dir = os.path.join(dataset_dir, "pointcloud_voxelgrid")
    else:
        if use_local:
            if use_normal:
                vg_or_pcl_dir = os.path.join(dataset_dir, "voxels_features_normals")
            else:
                vg_or_pcl_dir = os.path.join(dataset_dir, "voxels_features")
        else:
            if use_normal:
                vg_or_pcl_dir = os.path.join(dataset_dir, "voxels_xyz_normals_features")
            else:
                vg_or_pcl_dir = os.path.join(dataset_dir, "voxels_xyz_features")

    sceneflow_dir = os.path.join(dataset_dir, "sceneflow")
    samples = load_files(vg_or_pcl_dir, sceneflow_dir, data_split, data_type, sequences_to_eval)

    ## Create TRAIN loader
    eval_dataset = Dataset(samples)
    print("eval Dataset's length:", len(eval_dataset))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=8, collate_fn=detection_collate,
                                              drop_last=True, pin_memory=False)


    print("Number of eval batches: ", len(eval_loader),
          "(Samples: ", str(len(eval_loader) * batch_size), ")")

    return eval_loader


#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################


def compute_voxelgrid_and_sceneflow(color_frame, of_frame, disp_frame, dispChange_frame,
                                    data_type_3D):

    # import time
    # import matplotlib.pyplot as plt
    # import cv2
    height, width, _ = color_frame.shape

    ## Store our input data with high precision
    # colors_np_A = color_frame.reshape(-1, 3)
    of = np.asarray(of_frame, dtype=np.float64)
    disp = np.asarray(disp_frame, dtype=np.float64)
    dispChange = np.asarray(dispChange_frame, dtype=np.float64)

    ## Create our matrix of indices
    indices = np.indices((height, width))
    py, px = indices[0], indices[1]

    ## Get 3D Point Cloud
    z = np.float64(cfg.baseline) * np.float64(cfg.fx) / disp
    x = np.multiply((px - np.float64(cfg.cx)), z) / np.float64(cfg.fx)
    y = np.multiply((py - np.float64(cfg.cy)), z) / np.float64(cfg.fy)
    coordinates_np_matrix = np.dstack((x, y, z))
    coordinates_np_matrix_cropped = coordinates_np_matrix[1:-1, 1:-1]
    coordinates_np = coordinates_np_matrix_cropped.reshape(-1, 3)

    ## Normal map
    A = coordinates_np_matrix[2:, 1:-1] - coordinates_np_matrix[0:-2, 1:-1]
    B = coordinates_np_matrix[1:-1, 2:] - coordinates_np_matrix[1:-1, 0:-2]
    normal_matrix = np.cross(A, B, axis=2)
    norm = np.linalg.norm(normal_matrix, axis=2)
    normal_matrix[:, :, 0] /= norm
    normal_matrix[:, :, 1] /= norm
    normal_matrix[:, :, 2] /= norm
    normal_np = normal_matrix.reshape(-1, 3)
    ## For visualization
    # normal += 1
    # normal /= 2
    # cv2.imshow("normal", normal)
    # cv2.waitKey()
    # exit()
    ## For visualization

    ## Compute scene flow (by first getting optical flow from input)
    u = of[:, :, 0]  # Optical flow in horizontal direction
    v = of[:, :, 1]  # optical flow in vertical direction
    m = np.float64(cfg.baseline) / (disp + dispChange)
    dX = np.multiply(m, u - np.divide(np.multiply(dispChange, px - np.float64(cfg.cx)), disp))
    dY = np.multiply(m, v - np.divide(np.multiply(dispChange, py - np.float64(cfg.cy)), disp))
    dZ = cfg.fx * cfg.baseline * ((1.0 / (disp + dispChange)) - (1.0 / disp))
    sceneflow_np_matrix = np.dstack((dX, dY, dZ))
    sceneflow_np_matrix_cropped = sceneflow_np_matrix[1:-1, 1:-1]
    sceneflow_np = sceneflow_np_matrix_cropped.reshape(-1, 3)

    if data_type_3D == DATA_TYPES_3D['POINTCLOUD']:
        mask = coordinates_np[:, 2] <= cfg.max_z
        coordinates_np = coordinates_np[mask]
        normal_np = normal_np[mask]
        sceneflow_np = sceneflow_np[mask]
        return (coordinates_np, normal_np), sceneflow_np

    points, normals, sceneflows = filter_pointcloud(coordinates_np, normal_np, sceneflow_np)

    if points.size == 0:
        return None, None

    voxel_coords = ((points - np.array([cfg.xrange[0], cfg.yrange[0], cfg.zrange[0]]))
                    / (cfg.vx, cfg.vy, cfg.vz)).astype(np.int32)

    voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
                                                    return_inverse=True, return_counts=True)
    ## NOTE: inv_ind (inverse indices) : for every point, the voxel index in which the point resides

    ## TODO: REMOVE VOXELS WHICH CONTAIN LESS THAN A CERTAIN NUMBER OF POINTS ##
    # voxel_coords = voxel_coords[voxel_counts >= cfg.t]
    # good_pts_mask = get_good_pts_mask(voxel_counts, inv_ind, len(points))
    # points = points[good_pts_mask]
    # normals = normals[good_pts_mask]
    # sceneflows = sceneflows[good_pts_mask]
    # inv_ind = inv_ind[good_pts_mask]
    ## TODO: REMOVE VOXELS WHICH CONTAIN LESS THAN A CERTAIN NUMBER OF POINTS ##

    voxel_sceneflows = []
    # max_pts_inv_ind = []
    # voxel_pts_ind = []
    for i in range(len(voxel_coords)):
        mask = inv_ind == i
        sfs = sceneflows[mask]
        # pts_global_ind = np.asarray(list(compress(range(len(mask)), mask)), dtype=np.int32)

        sfs = np.median(sfs, axis=0)
        voxel_sceneflows.append(sfs)
        # max_pts_inv_ind.append(inv_ind[pts_global_ind])
        # voxel_pts_ind.append(pts_global_ind)

    return (points, normals, voxel_coords, inv_ind, voxel_counts), (sceneflows, voxel_sceneflows)


def preprocess_pointcloud(points, normals, sample_name, sceneflows=None):
    pass
    # if (points.size == 0):
    #     raise Exception(sample_name, "has no points with current ranges!")
    #
    # points, normals, sceneflows = filter_pointcloud(points, normals, sceneflows)
    # points, normals, sceneflows = randomize(points, normals, sceneflows)
    #
    # voxel_coords = ((points - np.array([cfg.xrange[0], cfg.yrange[0], cfg.zrange[0]]))
    #                 / (cfg.vx, cfg.vy, cfg.vz)).astype(np.int32)
    #
    # voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
    #                                                 return_inverse=True, return_counts=True)
    #
    # voxel_features = []
    # voxel_sceneflows = [] if sceneflows is not None else None
    # # max_pts_inv_ind = [] if sceneflows is not None else None
    # # voxel_pts_ind = [] if sceneflows is not None else None
    # for i in range(len(voxel_coords)):
    #     voxel = np.zeros((cfg.T, cfg.f), dtype=np.float64)
    #     mask = inv_ind == i
    #     n = normals[mask]
    #     sfs = sceneflows[mask] if sceneflows is not None else None
    #     # pts_global_ind = np.asarray(list(compress(range(len(mask)), mask)), dtype=np.int32)
    #
    #     if voxel_counts[i] > cfg.T:
    #         pts = pts[:cfg.T, :]
    #         n = n[:cfg.T, :]
    #         sfs = sfs[:cfg.T, :] if sceneflows is not None else None
    #         # pts_global_ind = pts_global_ind[:cfg.T]
    #
    #     ## augment the points with their coordinate in the voxel's reference system
    #     voxel[:pts.shape[0], :] = np.concatenate((pts, pts - centroid(pts), n), axis=1)
    #     voxel_features.append(voxel)
    #
    #     if sceneflows is not None:
    #         sfs = np.median(sfs, axis=0)
    #         voxel_sceneflows.append(sfs)
    #         # max_pts_inv_ind.append(inv_ind[pts_global_ind])
    #         # voxel_pts_ind.append(pts_global_ind)
    #
    # if sceneflows is not None:
    #     voxel_sceneflows = np.array(voxel_sceneflows)
    #     # max_pts_inv_ind = np.concatenate(max_pts_inv_ind)
    #     # voxel_pts_ind = np.concatenate(voxel_pts_ind)
    #
    # return (points, np.array(voxel_features), voxel_coords, inv_ind), \
    #        (sceneflows, voxel_sceneflows)


def preprocess_pointnet(points, normals, voxel_coords, inv_ind, sceneflows=None):

    points, normals, inv_ind, sceneflows = randomize(points, normals, inv_ind, sceneflows)

    voxel_pts = []
    voxel_features = []
    for i in range(len(voxel_coords)):
        pts_np = np.zeros((cfg.T, 3), dtype=np.float64)
        normals_np = np.zeros((cfg.T, 3), dtype=np.float64)

        mask = inv_ind == i
        pts = points[mask]
        n = normals[mask]
        sfs = sceneflows[mask] if sceneflows is not None else None

        if pts.shape[0] > cfg.T:
            pts = pts[:cfg.T, :]
            n = n[:cfg.T, :]
            sfs = sfs[:cfg.T, :] if sceneflows is not None else None

        pts_np[:pts.shape[0], :] = pts
        normals_np[:pts.shape[0], :] = n
        voxel_pts.append(pts_np)
        voxel_features.append(normals_np)

    return (points, np.array(voxel_pts), np.array(voxel_features), voxel_coords, inv_ind), \
           sceneflows

def preprocess_voxelnet(points, normals, voxel_coords, inv_ind, sceneflows=None):

    points, normals, inv_ind, sceneflows = randomize(points, normals, inv_ind, sceneflows)

    voxel_features = []
    for i in range(len(voxel_coords)):
        voxel = np.zeros((cfg.T, 9), dtype=np.float32)
        mask = inv_ind == i
        pts = points[mask]
        n = normals[mask]

        if pts.shape[0] > cfg.T:
            pts = pts[:cfg.T, :]
            n = n[:cfg.T, :]

        voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - centroid(pts), n), axis=1)
        voxel_features.append(voxel)

    return (points, np.array(voxel_features), voxel_coords, inv_ind), sceneflows


def get_good_pts_mask(voxel_counts, inv_ind, n_pts):
    ######################################################################
    ## REMOVE VOXELS WHICH CONTAIN LESS THAN A CERTAIN NUMBER OF POINTS ##
    ############## AND REMOVE ALSO THE CORRESPONDING POINTS ##############
    ## Get the indices of those bad voxels
    bad_voxels_ind = np.where(voxel_counts < cfg.t)[0]
    ## Compute the indices of the points contained in those bad voxels
    bad_pts_ind = np.concatenate([np.nonzero(inv_ind == bad)[0] for bad in bad_voxels_ind])
    ## Create a mask for the good points
    good_pts_mask = np.ones(n_pts, dtype=bool)
    good_pts_mask[bad_pts_ind] = False
    return good_pts_mask


def centroid(pts):
    length = pts.shape[0]
    sum_x = np.sum(pts[:, 0])
    sum_y = np.sum(pts[:, 1])
    sum_z = np.sum(pts[:, 2])
    return np.array([sum_x/length, sum_y/length, sum_z/length])


def compute_PCA(pts):
    pca = PCA(n_components=3)
    pca.fit(pts)
    pca_score = pca.explained_variance_ratio_
    V = pca.components_
    # x_pca_axis, y_pca_axis, z_pca_axis = 0.2 * V
    normal_vector = V[np.argmin(pca_score, axis=0)]

    ## VISUALIZE STUFF ##
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # centr = centroid(pts)
    # fig = plt.figure(1, figsize=(4, 3))
    # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=20)
    # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='+', alpha=.4)
    # ax.quiver(centr[0], centr[1], centr[2], x_pca_axis[0], x_pca_axis[1], x_pca_axis[2], color='r')
    # ax.quiver(centr[0], centr[1], centr[2], y_pca_axis[0], y_pca_axis[1], y_pca_axis[2], color='g')
    # ax.quiver(centr[0], centr[1], centr[2], normal_vector[0], normal_vector[1], normal_vector[2], color='b')
    # plt.show()
    ## VISUALIZE STUFF ##

    return normal_vector


def filter_pointcloud(points, normals, sceneflows=None):
    pxs = points[:, 0]
    pys = points[:, 1]
    pzs = points[:, 2]

    filter_x = np.where((pxs >= cfg.xrange[0]) & (pxs < cfg.xrange[1]))[0]
    filter_y = np.where((pys >= cfg.yrange[0]) & (pys < cfg.yrange[1]))[0]
    filter_z = np.where((pzs >= cfg.zrange[0]) & (pzs < cfg.zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)

    if sceneflows is not None:
        sceneflows = sceneflows[filter_xyz]

    return points[filter_xyz], normals[filter_xyz], sceneflows


def randomize(points, normals, inv_ind, sceneflows=None):
    if sceneflows is not None:
        assert points.shape==normals.shape==sceneflows.shape, "randomize 1 "
    else:
        assert points.shape==normals.shape, "Inputs with different shapes in randomize 1"

    assert points.shape[0] == len(inv_ind), "Inputs with different shapes in randomize 2"

    # Generate the permutation index array.
    permutation = np.random.permutation(points.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_points = points[permutation]
    shuffled_normals = normals[permutation]
    shuffled_inv_ind = inv_ind[permutation]
    shuffled_sceneflows = sceneflows[permutation] if sceneflows is not None else None
    return shuffled_points, shuffled_normals, shuffled_inv_ind, shuffled_sceneflows


#######################
## COLLATE FUNCTIONS ##
#######################
def detection_collate_baseline_train(batch):
    voxel_coords_t0 = []

    voxel_coords_t1 = []

    voxel_sceneflows = []
    sample_names = []

    for i, sample in enumerate(batch):
        # Pointcloud data t0
        voxel_coords_t0.append(
            np.pad(sample[0], ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

        # Pointcloud data t1
        voxel_coords_t1.append(
            np.pad(sample[1], ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

        voxel_sceneflows.append(sample[2])
        sample_names.append(sample[3])

    return np.concatenate(voxel_coords_t0), \
           np.concatenate(voxel_coords_t1), \
           np.concatenate(voxel_sceneflows), \
           sample_names

def detection_collate_baseline_test(batch):
    points_t0 = []
    voxel_coords_t0 = []
    inv_ind_t0 = []

    points_t1 = []
    voxel_coords_t1 = []

    sceneflows = []
    voxel_sceneflows = []
    sample_names = []

    for i, sample in enumerate(batch):
        # Pointcloud data t0
        points_t0.append(sample[0][0])
        voxel_coords_t0.append(
            np.pad(sample[0][1], ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))
        inv_ind_t0.append(sample[0][2])

        # Pointcloud data t1
        points_t1.append(sample[1][0])
        voxel_coords_t1.append(
            np.pad(sample[1][1], ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

        sceneflows.append(sample[2][0])
        voxel_sceneflows.append(sample[2][1])
        sample_names.append(sample[3])

    return (np.concatenate(points_t0), np.concatenate(voxel_coords_t0),
            np.concatenate(inv_ind_t0)), \
           (np.concatenate(points_t1), np.concatenate(voxel_coords_t1),
            None), \
           (np.concatenate(sceneflows), np.concatenate(voxel_sceneflows)), \
           sample_names


def detection_collate_pointnet_train(batch):
    voxels_features_t0 = []
    voxels_coords_t0 = []

    voxels_features_t1 = []
    voxels_coords_t1 = []

    voxels_sceneflows = []
    sample_names = []

    for i, sample in enumerate(batch):
        # Pointcloud data t0
        voxels_features_t0.append(sample[0][0])
        voxels_coords_t0.append(
            np.pad(sample[0][1], ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

        # Pointcloud data t1
        voxels_features_t1.append(sample[1][0])
        voxels_coords_t1.append(
            np.pad(sample[1][1], ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

        voxels_sceneflows.append(sample[2])
        sample_names.append(sample[3])

    return (np.concatenate(voxels_features_t0), np.concatenate(voxels_coords_t0)), \
           (np.concatenate(voxels_features_t1), np.concatenate(voxels_coords_t1)), \
           np.concatenate(voxels_sceneflows), sample_names


def detection_collate_pointnet_test(batch):
    points_t0 = []
    voxels_features_t0 = []
    voxels_coords_t0 = []
    inv_ind_t0 = []

    points_t1 = []
    voxels_features_t1 = []
    voxels_coords_t1 = []

    sceneflows = []
    voxels_sceneflows = []
    sample_names = []

    for i, sample in enumerate(batch):
        # Pointcloud data t0
        points_t0.append(sample[0][0])
        voxels_features_t0.append(sample[0][1])
        voxels_coords_t0.append(
            np.pad(sample[0][2], ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))
        inv_ind_t0.append(sample[0][3])

        # Pointcloud data t1
        points_t1.append(sample[1][0])
        voxels_features_t1.append(sample[1][1])
        voxels_coords_t1.append(
            np.pad(sample[1][2], ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

        sceneflows.append(sample[2][0])
        voxels_sceneflows.append(sample[2][1])
        sample_names.append(sample[3])

    return (np.concatenate(points_t0), np.concatenate(voxels_features_t0),
            np.concatenate(voxels_coords_t0), np.concatenate(inv_ind_t0)), \
           (np.concatenate(points_t1), np.concatenate(voxels_features_t1),
            np.concatenate(voxels_coords_t1), None), \
           (np.concatenate(sceneflows), np.concatenate(voxels_sceneflows)), \
           sample_names


#######################
## COLLATE FUNCTIONS ##
#######################


def generate_numpy(pcl_dir, sf_dir, frame_path, pointcloud_data, scenflow_data, data_type_3D):
    """
    :param vg_dir:
    :param sf_gt_dir:
    :param frame_path:
    :param pointcloud_data:
    :return:
    """

    base_name = os.path.basename(frame_path)
    base_name = os.path.splitext(base_name)[0]

    pcl_name = pcl_dir + "/" + base_name
    sceneflow_name = sf_dir + "/" + base_name

    if data_type_3D == DATA_TYPES_3D['POINTCLOUD']:
        if pointcloud_data is None:
            print(base_name, "----No points in", pcl_name, "- Not saving neither points nor sceneflow")
            remove_corresponding("from_point", base_name, pcl_dir, sf_dir, pcl_ext="npz", sf_ext="npy")
            return

        coordinates_np, normal_np = pointcloud_data
        sceneflow_np = scenflow_data
        assert coordinates_np.shape == normal_np.shape == sceneflow_np.shape, \
            "Vertices and Sceneflow have different shapes"

        ## Points
        if data_is_corrupted(coordinates_np):
            print(base_name, "----Points in", pcl_name, "are corrupted. Not saving neither points nor sceneflow")
            remove_corresponding("from_point", base_name, pcl_dir, sf_dir, pcl_ext="npz", sf_ext="npy")
            return
        np.savez(pcl_name, coordinates_np=coordinates_np, normal_np=normal_np, allow_pickle=False)

        ## Sceneflow
        if not base_name == "0015":
            if data_is_corrupted(sceneflow_np):
                print(base_name, "----Sceneflow in", sceneflow_name, "is corrupted. Not saving sceneflow")
                remove_corresponding("from_sceneflow", base_name, pcl_dir, sf_dir, pcl_ext="npz", sf_ext="npy")
                return
            np.save(sceneflow_name, sceneflow_np, allow_pickle=False)

    elif data_type_3D == DATA_TYPES_3D['BOTH']:
        if pointcloud_data is None:
            print(base_name, "----No points in", pcl_name, "- Not saving neither points nor sceneflow")
            remove_corresponding("from_point", base_name, pcl_dir, sf_dir, pcl_ext="npz", sf_ext="npz")
            return

        points, normals, voxel_coords, inv_ind, voxel_counts = pointcloud_data
        sceneflow_np, voxel_sceneflows = scenflow_data
        assert points.shape == sceneflow_np.shape, "Points and Sceneflow have different shapes"

        ## Points
        if data_is_corrupted(points) or data_is_corrupted(normals):
            print(base_name, "----Points in", pcl_name, "are corrupted. Not saving neither points nor sceneflow")
            remove_corresponding("from_point", base_name, pcl_dir, sf_dir, pcl_ext="npz", sf_ext="npz")
            return
        np.savez(pcl_name, points=points, normals=normals, voxel_coords=voxel_coords,
                 inv_ind=inv_ind, voxel_counts=voxel_counts,
                 allow_pickle=False)

        ## Sceneflow
        if not base_name == "0015":
            if data_is_corrupted(sceneflow_np):
                print(base_name, "----Sceneflow in", sceneflow_name, "is corrupted. Not saving sceneflow")
                remove_corresponding("from_sceneflow", base_name, pcl_dir, sf_dir, pcl_ext="npz", sf_ext="npz")
                return
            np.savez(sceneflow_name, sceneflow_np=sceneflow_np, voxel_sceneflows=voxel_sceneflows, allow_pickle=False)


def data_is_corrupted(data):
    if data.size == 0:
        return True
    number_of_nonNaN = np.count_nonzero(~np.isnan(data))
    if data.size - number_of_nonNaN != 0:
        return True
    return False


def remove_corresponding(from_what, base_name, pcl_dir, sf_dir, pcl_ext, sf_ext):
    if from_what == "from_point":
        if not base_name == "0006":
            prev_base_name = str(int(base_name) - 1).zfill(4)
            print("--------Will try to remove sceneflow of", prev_base_name)
            prev_sf_name = sf_dir + "/" + prev_base_name + "." + sf_ext
            if os.path.exists(prev_sf_name):
                print(Fore.GREEN + "--------Removal of sf", prev_base_name, "succesful" + Style.RESET_ALL)
                os.remove(prev_sf_name)
            else:
                print(Fore.RED + "--------Removal not successful" + Style.RESET_ALL)

            if prev_base_name == "0006":
                print("--------Will also try to remove pointcloud of", prev_base_name)
                prev_pcl_name = pcl_dir + "/" + prev_base_name + "." + pcl_ext
                if os.path.exists(prev_pcl_name):
                    print(Fore.GREEN + "--------Removal of pcl", prev_base_name, "succesful" + Style.RESET_ALL)
                    os.remove(prev_pcl_name)
                else:
                    print(Fore.RED + "--------Removal not successful" + Style.RESET_ALL)

    elif from_what == "from_sceneflow":
        if base_name == "0006":
            pcl_name = pcl_dir + "/" + base_name + "." + pcl_ext
            print("--------Will try to remove pcl of", pcl_name)
            if os.path.exists(pcl_name):
                print(Fore.GREEN + "--------Removing as well pointcloud", pcl_name)
                os.remove(pcl_name)
            else:
                print(Fore.RED + "--------Removal not successful" + Style.RESET_ALL)

def compute_rotation_to_align_vectors(vector):
    """
    Compute R that aligns the given vector with
    the direction vector of our arrow ply file [0, 0, 1]
    :param vector:
    :return:
    """
    ## Normalize vector
    vector = vector / np.linalg.norm(vector)

    ## The arrow from the ply file always looks along the z axis
    arrow_vector = np.zeros_like(vector)
    arrow_vector[2] = 1.0

    v = np.cross(arrow_vector, vector)
    c = np.dot(vector, arrow_vector)
    s = np.linalg.norm(v)
    I = np.identity(3)
    vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
    skew = np.matrix(vXStr)

    R = I + skew + (np.matmul(skew, skew) / (1 + c))

    det = np.linalg.det(R)

    if (1 - det > cfg.rot_matrix_threshold):
        raise Exception("Not a rotation matrix...", det)

    return R


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def end_point_error(sceneflow_pred_np, sceneflow_gt_np):
    return np.mean(np.sqrt(np.sum((sceneflow_pred_np - sceneflow_gt_np)**2, axis=-1)))


def angular_error(flow1, flow2):
    """returns the angular error between two flow fields.
    Parameters
    ----------
    flow1 : ndarray.
        First optical flow field.
    flow2 : ndarray.
        Second optical flow field.
    Returns
    -------
    AE : angular error field.
        Scalar field with the angular error field in degrees.
    """

    f1_x = flow1[..., 0]
    f1_y = flow1[..., 1]
    f1_z = flow1[..., 2]

    f2_x = flow2[..., 0]
    f2_y = flow2[..., 1]
    f2_z = flow2[..., 2]

    top = 1.0 + (f1_x * f2_x) + (f1_y * f2_y) + (f1_z * f2_z)
    bottom = np.sqrt(1.0 + (f1_x * f1_x) + (f1_y * f1_y) + (f1_z * f1_z)) * \
             np.sqrt(1.0 + (f2_x * f2_x) + (f2_y * f2_y) + (f2_z * f2_z))

    return np.rad2deg(np.arccos(top / bottom))


def metrics(pred, gt):
    error = np.sqrt(np.sum((pred - gt)**2, axis=-1) + 1e-20)
    EPE = np.mean(error)

    acc1 = np.sum(error < 0.05)
    acc1 = acc1 / len(error)

    acc2 = np.sum(error < 0.10)
    acc2 = acc2 / len(error)

    return EPE, acc1, acc2
