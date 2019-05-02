import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
np.set_printoptions(threshold=np.nan)
import os, shutil
import glob
import platform
import copy
import time

from utils import read, compute_voxelgrid_and_sceneflow, generate_numpy, create_dir
from PlyFile import PlyFile
from config import Config, DATA_TYPES_3D

##################################################################################
system = platform.system()
cfg = Config(system=system)
print("Config init from ", cfg.dataset_path)
##################################################################################

# generate = input("Generate numpy files? ")
# if generate == "Y" or generate == "y":
#     generate = True
# elif generate == "F" or generate == "f":
#     generate = False
#
# print(generate)

SPLIT = "TRAIN"
LETTER = "B"
NUMBER = "0740"

data_type_3D = cfg.data_type_3d

generate = False
just_one_sample = False
if generate:

    splits = ["TRAIN", "TEST"]
    data_folders = []

    for split in splits:
        if split != SPLIT and just_one_sample:
            continue
        for letter in os.listdir(os.path.join(cfg.dataset_path, "frames_cleanpass", split)):
            start = time.time()
            if letter != LETTER and just_one_sample:
                continue
            for number in os.listdir(os.path.join(cfg.dataset_path, "frames_cleanpass", split, letter)):
                if number != NUMBER and just_one_sample:
                    continue
                data_folder = os.path.join(split, letter, number)
                print("Processing", os.path.join(cfg.dataset_path, "frames_cleanpass", data_folder, "left"))

                ## Get PATHS of the COLOR frames
                color_frame_paths = []
                files = glob.glob(os.path.join(cfg.dataset_path, "frames_cleanpass", data_folder, "left", "*"))
                for color_frame_path in sorted(files):
                    color_frame_paths.append(color_frame_path.replace('\\', '/'))

                ## Get PATHS of the OPTICAL FLOW frames
                of_frame_paths = []
                files = glob.glob(os.path.join(cfg.dataset_path, "optical_flow", data_folder, cfg.into, "left", "*"))
                for of_frame_path in sorted(files):
                    of_frame_paths.append(of_frame_path.replace('\\', '/'))

                ## Get PATHS of the DISPARITY frames
                disp_frame_paths = []
                files = glob.glob(os.path.join(cfg.dataset_path, "disparity", data_folder, "left", "*"))
                for disp_frame_path in sorted(files):
                    disp_frame_paths.append(disp_frame_path.replace('\\', '/'))

                # Get PATHS of the DISPARITY CHANGE frames
                dispChange_frame_paths = []
                files = glob.glob(os.path.join(cfg.dataset_path, "disparity_change", data_folder, cfg.into, "left", "*"))
                for dipsChange_frame_path in sorted(files):
                    dispChange_frame_paths.append(dipsChange_frame_path.replace('\\', '/'))

                assert len(color_frame_paths) == len(of_frame_paths) == len(disp_frame_paths) == len(dispChange_frame_paths)
                # print(len(color_frame_paths), len(of_frame_paths), len(disp_frame_paths), len(dispChange_frame_paths))
                if len(color_frame_paths) == 0:
                    raise Exception("No files were loaded!")

                ##################################################################################

                # Get the color frames
                color_frames = []
                for color_frame_path in color_frame_paths:
                    color_frames.append(read(color_frame_path))

                # Get the optical flow frames
                of_frames = []
                for of_frame_path in of_frame_paths:
                    of_frame = read(of_frame_path)
                    of_frames.append(of_frame)

                # Get the depth frames
                disp_frames = []
                for disp_frame_path in disp_frame_paths:
                    disp_frame = read(disp_frame_path)
                    disp_frames.append(disp_frame)

                # Get the depth change frames
                dispChange_frames = []
                for i, dispChange_frame_path in enumerate(dispChange_frame_paths):
                    dispChange_frame = read(dispChange_frame_path)
                    dispChange_frames.append(dispChange_frame)

                assert len(color_frames) == len(of_frames) == len(disp_frames) == len(dispChange_frames)
                if len(color_frames) == 0:
                    raise Exception("Could not read files!")

                # plt.imshow(disp_frames[1])
                # plt.colorbar()
                # plt.show()
                # plt.imshow(dispChange_frames[1])
                # plt.colorbar()
                # plt.show()
                # plt.imshow(depthChange_frames[1])
                # plt.colorbar()
                # plt.scatter(325, 270)
                # plt.show()
                #
                # print(depthChange_frames[1][400][400])
                # print(depthChange_frames[1][280][320])

                ##################################################################################
                ##################################################################################

                ## Generate numpy arrays from raw data and store it
                ## At this point we have the paths, let's compute the numpy files

                # Prepare dir names
                if cfg.system == 'Windows':
                    if data_type_3D == DATA_TYPES_3D['POINTCLOUD']:
                        ## POINTCLOUD ##
                        pointcloud_dir = os.path.join(cfg.root_dir, cfg.np_datasets_base_path, cfg.dataset,
                                                      "pointcloud", "pointcloud", split, letter, number)
                        sceneflow_dir = os.path.join(cfg.root_dir, cfg.np_datasets_base_path, cfg.dataset,
                                                     "pointcloud", "sceneflow", split, letter, number)

                    elif data_type_3D == DATA_TYPES_3D['BOTH']:
                        ## BOTH ##
                        dim_str = "{}_{}_{}_{}_{}_{}".format(abs(cfg.xrange[0]), cfg.xrange[1],
                                                             abs(cfg.yrange[0]), cfg.yrange[1],
                                                             abs(cfg.zrange[0]), cfg.zrange[1])
                        pointcloud_dir = os.path.join(cfg.root_dir, cfg.np_datasets_base_path, cfg.dataset,
                                                      "pointcloud_voxelgrid_"+dim_str, "pointcloud_voxelgrid",
                                                      split, letter, number)
                        sceneflow_dir = os.path.join(cfg.root_dir, cfg.np_datasets_base_path, cfg.dataset,
                                                     "pointcloud_voxelgrid_"+dim_str, "sceneflow",
                                                     split, letter, number)

                elif cfg.system == 'Linux':
                    if data_type_3D == DATA_TYPES_3D['POINTCLOUD']:
                        ## POINTCLOUD ##
                        pcl_dir = os.path.join(cfg.root_dir, cfg.np_datasets_base_path, cfg.dataset,
                                              "pointcloud", "pointcloud", split, letter, number)
                        sceneflow_dir = os.path.join(cfg.root_dir, cfg.np_datasets_base_path, cfg.dataset,
                                                 "pointcloud", "sceneflow", split, letter, number)

                create_dir(pointcloud_dir)
                create_dir(sceneflow_dir)

                for i in range(len(disp_frame_paths)):
                    # print("----", i)
                    ## Generate point cloud ( [colors, coordinates, sceneflow] ) as panda's DataFrame
                    pointcloud_data, sceneflow_data = compute_voxelgrid_and_sceneflow(color_frames[i],
                                                                                      of_frames[i], disp_frames[i],
                                                                                      dispChange_frames[i], data_type_3D)

                    ## Store as numpy array
                    generate_numpy(pointcloud_dir, sceneflow_dir, color_frame_paths[i],
                                   pointcloud_data, sceneflow_data, data_type_3D)

                try:
                    os.rmdir(pointcloud_dir)
                    os.rmdir(sceneflow_dir)
                except:
                    pass

            print("Time dedicated to current LETER was:", time.time() - start)

else:
    ### Read numpy files and work with them ###

    dataset = cfg.dataset

    if data_type_3D == DATA_TYPES_3D['POINTCLOUD']:
        pass
        # ## First, get the voxelgrids and sceneflow paths
        # pcl_dir = os.path.join(cfg.data_pcl_dir, "pointcloud", SPLIT, LETTER, NUMBER)
        # sf_pcl_dir = os.path.join(cfg.data_pcl_dir, "sceneflow", SPLIT, LETTER, NUMBER)
        #
        # paths = []
        # for path in sorted(glob.glob(pcl_dir + "/*")):
        #     paths.append(path.replace('\\', '/'))
        #
        # sceneflow_paths = []
        # for path in sorted(glob.glob(sf_pcl_dir + "/*")):
        #     sceneflow_paths.append(path.replace('\\', '/'))
        #
        # assert len(sceneflow_paths) != 0

    elif data_type_3D == DATA_TYPES_3D['BOTH']:
        ## First, get the voxelgrids and sceneflow paths
        pcl_data_dir = os.path.join(cfg.data_dir[cfg.data_type_3d], "pointnet_data", SPLIT, LETTER, NUMBER)
        sf_data_dir = os.path.join(cfg.data_dir[cfg.data_type_3d], "sceneflow", SPLIT, LETTER, NUMBER)

        print(pcl_data_dir)

        paths = []
        for path in sorted(glob.glob(pcl_data_dir + "/*")):
            paths.append(path.replace('\\', '/'))

        sceneflow_paths = []
        for path in sorted(glob.glob(sf_data_dir + "/*")):
            sceneflow_paths.append(path.replace('\\', '/'))

    ########################################################
    for i in range(len(paths) - 1):

        if cfg.data_type_3d == DATA_TYPES_3D['POINTCLOUD']:
            pass
        elif cfg.data_type_3d == DATA_TYPES_3D['BOTH']:
            pcl_data_t0 = np.load(paths[i])
            sf_data_t0 = np.load(sceneflow_paths[i])

            points = pcl_data_t0['points']
            plyfile = PlyFile(points, color=[255, 255, 0])
            if cfg.system == 'Windows':
                file_name = os.path.splitext(paths[i])[0].split('/')
                file_name = file_name[-4] + "-" + file_name[-3] + "-" + file_name[-2] + "-" + file_name[-1]
                plyfile.write_ply(cfg.data_dir[cfg.data_type_3d] + "/vg_" + file_name + ".ply")

        #points

        # print(i, "Reading")
        # ## Read one input
        # vertices = np.load(paths[i])
        # vertices_raw = copy.deepcopy(vertices)
        # if data_type_3D == DATA_TYPES_3D['VOXELGRID']:
        #     vertices = np.argwhere(vertices)
        #
        # ## Read the corresponding sceneflow voxelgrid groundtruth
        # sceneflow_vertices = np.load(sceneflow_paths[i])
        # if data_type_3D == DATA_TYPES_3D['VOXELGRID']:
        #     sceneflow_vertices = sceneflow_vertices[np.where(vertices_raw == 1)]
        #
        # ########################################################
        #
        # ## Get our arrow
        # arrow_vertices, arrow_faces = PlyFile.read_ply("ply_examples/awesome_rectangular_arrow.ply")
        #
        # ########################################################
        # ## Generate a PlyFile object and initialize it with our points from the voxelgrid
        # plyfile = PlyFile(vertices, color=[255, 255, 0])
        #
        # # plyfile.draw_arrows_for_sceneflow(cfg, vertices, sceneflow_vertices, arrow_vertices, arrow_faces)
        #
        # file_name = os.path.splitext(paths[i])[0].split('/')
        # file_name = file_name[-4] + "-" + file_name[-3] + "-" + file_name[-2] + "-" + file_name[-1]
        # if cfg.system == 'Windows':
        #     if data_type_3D == DATA_TYPES_3D['VOXELGRID']:
        #         plyfile.write_ply(cfg.data_vg_dir + "/vg_" + file_name + ".ply")
        #     elif data_type_3D == DATA_TYPES_3D['POINTCLOUD']:
        #         print(cfg.data_pcl_dir + "/pointcloud_" + file_name + ".ply")
        #         plyfile.write_ply(cfg.data_pcl_dir + "/pointcloud_" + file_name + ".ply")
        # elif cfg.system == 'Linux':
        #     if data_type_3D == DATA_TYPES_3D['VOXELGRID']:
        #         plyfile.write_ply("/mnt/raid/pablo/experiments/quick_tests/vg" + str(cfg.n_voxels) + "_"
        #                              + str(cfg.threshold) + "crop_" + file_name + ".ply")
        #     elif data_type_3D == DATA_TYPES_3D['POINTCLOUD']:
        #         plyfile.write_ply("/mnt/raid/pablo/experiments/quick_tests/pointcloud" + file_name + ".ply")
