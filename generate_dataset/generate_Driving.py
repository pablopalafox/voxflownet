import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
#np.set_printoptions(threshold=np.nan)
import os, shutil
import glob
import platform

from utils import read, compute_voxelgrid_and_sceneflow, generate_numpy, PlyFile, create_dir
from config import Config

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

generate = True
if generate:

    if cfg.dataset == 'FlyingThings3D':
        data_folder_split = "TRAIN"
        letter = "A"
        data_folders = []
    elif cfg.dataset == 'Driving':
        focallength = "15mm_focallength"
        direction = "scene_forwards"
        speed = "slow"
        cam = "left"

    ## Get the "number" folders under the "letter" folder set above
    ## Since it's the same distribution for every feature (disparity, optical_clow, ...)
    ## we can just check one

    for nubmer in os.listdir(os.path.join(cfg.dataset_path, "frames_cleanpass", data_folder_split, letter)):
        data_folders.append(os.path.join(data_folder_split, letter, nubmer))


    max_n_frames = -1
    n_frames = 0

    for data_folder in data_folders:
        print(data_folder)

        ## Get PATHS of the COLOR frames
        color_frame_paths = []
        if cfg.dataset == 'FlyingThings3D':
            for data_folder in data_folders:

                files = glob.glob(os.path.join(cfg.dataset_path, "frames_cleanpass", data_folder, "left", "*"))
        elif cfg.dataset == 'Driving':
            files = glob.glob(os.path.join(cfg.dataset_path, "frames_cleanpass", focallength, direction, speed, cam, "*"))
        for color_frame_path in sorted(files):
            if n_frames == max_n_frames:
                break
            color_frame_paths.append(color_frame_path.replace('\\', '/'))
            n_frames += 1

        ## Get PATHS of the OPTICAL FLOW frames
        n_frames = 0
        of_frame_paths = []
        if cfg.dataset == 'FlyingThings3D':
            files = glob.glob(os.path.join(cfg.dataset_path, "optical_flow", data_folder, cfg.into, "left", "*"))
        elif cfg.dataset == 'Driving':
            files = glob.glob(os.path.join(cfg.dataset_path, "optical_flow", "15mm_focallength", "scene_forwards",
                                           "slow", cfg.into, "left", "*"))
        for of_frame_path in sorted(files):
            if n_frames == max_n_frames:
                break
            of_frame_paths.append(of_frame_path.replace('\\', '/'))
            n_frames += 1

        ## Get PATHS of the DISPARITY frames
        n_frames = 0
        disp_frame_paths = []
        if cfg.dataset == 'FlyingThings3D':
            files = glob.glob(os.path.join(cfg.dataset_path, "disparity", data_folder, "left", "*"))
        elif cfg.dataset == 'Driving':
            files = glob.glob(os.path.join(cfg.dataset_path, "disparity", "15mm_focallength", "scene_forwards",
                                           "slow", "left", "*"))
        for disp_frame_path in sorted(files):
            if n_frames == max_n_frames:
                break
            disp_frame_paths.append(disp_frame_path.replace('\\', '/'))
            n_frames += 1

        # Get PATHS of the DISPARITY CHANGE frames
        n_frames = 0
        dispChange_frame_paths = []
        if cfg.dataset == 'FlyingThings3D':
            files = glob.glob(os.path.join(cfg.dataset_path, "disparity_change", data_folder, cfg.into, "left", "*"))
        elif cfg.dataset == 'Driving':
            files = glob.glob(os.path.join(cfg.dataset_path, "disparity_change", "15mm_focallength", "scene_forwards",
                                           "slow", cfg.into,"left", "*"))
        for dipsChange_frame_path in sorted(files):
            if n_frames == max_n_frames:
                break
            dispChange_frame_paths.append(dipsChange_frame_path.replace('\\', '/'))
            n_frames += 1

        assert len(color_frame_paths) == len(of_frame_paths) == len(disp_frame_paths) == len(dispChange_frame_paths)
        # print(len(color_frame_paths), len(of_frame_paths), len(disp_frame_paths), len(dispChange_frame_paths))
        if len(color_frame_paths) == 0:
            raise Exception("No files were loaded!")

        ##################################################################################

        # Get the color frames
        color_frames = []
        for color_frame_path in color_frame_paths:
            print(color_frame_path)
            color_frames.append(read(color_frame_path))

        # Get the optical flow frames
        of_frames = []
        for of_frame_path in of_frame_paths:
            of_frame = read(of_frame_path)
            of_frames.append(of_frame)

        # Get the depth frames
        disp_frames = []
        depth_frames = []
        for disp_frame_path in disp_frame_paths:
            disp_frame = read(disp_frame_path)
            disp_frames.append(disp_frame)
            depth_frame = cfg.baseline * cfg.fx / disp_frame
            depth_frames.append(depth_frame)

        # Get the depth change frames
        dispChange_frames = []
        depthChange_frames = []
        for i, dispChange_frame_path in enumerate(dispChange_frame_paths):
            dispChange_frame = read(dispChange_frame_path)
            dispChange_frames.append(dispChange_frame)
            depthChange_frame = ((cfg.baseline * cfg.fx / (disp_frames[i] + dispChange_frame)) - (cfg.baseline * cfg.fx / disp_frames[i]))
            depthChange_frames.append(depthChange_frame)

        assert len(color_frames) == len(of_frames) == len(depth_frames) == len(depthChange_frames)
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
        vg_dir = os.path.join("data", cfg.dataset, str(cfg.n_voxels), "voxelgrids")
        sf_vg_dir = os.path.join("data", cfg.dataset, str(cfg.n_voxels), "sceneflows_vg")

        create_dir(vg_dir)
        create_dir(sf_vg_dir)

        for i in range(len(disp_frame_paths)):
            print(i, "- Processing", disp_frame_paths[i])

            # Generate point cloud ( [colors, coordinates, sceneflow] ) as panda's DataFrame
            pointcloud_data = compute_voxelgrid_and_sceneflow(color_frames[i], of_frames[i],
                                                              depth_frames[i], depthChange_frames[i],
                                                              compute_sceneflow=True, plot=False)

            ## Store as numpy array
            generate_numpy(vg_dir, sf_vg_dir, color_frame_paths[i], pointcloud_data)


else:
    ### Read numpy files and work with them ###

    ## First, get the voxelgrids and sceneflow paths
    dataset = cfg.dataset
    vg_dir = os.path.join(cfg.val_dir, "voxelgrids")
    sf_vg_dir = os.path.join(cfg.val_dir, "sceneflows_vg")

    voxelgrid_paths = []
    for path in sorted(glob.glob(vg_dir + "/*")):
        voxelgrid_paths.append(path.replace('\\', '/'))

    sceneflow_vg_paths = []
    for path in sorted(glob.glob(sf_vg_dir + "/*")):
        sceneflow_vg_paths.append(path.replace('\\', '/'))

    ########################################################
    ## Read one voxelgrid
    voxelgrid = np.load(voxelgrid_paths[0])
    voxel_vertices = np.argwhere(voxelgrid)

    ## Read the corresponding sceneflow voxelgrid groundtruth
    sceneflow = np.load(sceneflow_vg_paths[0])
    sf_vg_vertices = sceneflow[np.where(voxelgrid == 1)]
    ########################################################

    ## Get our arrow
    arrow_vertices, arrow_faces = PlyFile.read_ply("ply_examples/awesome_arrow.ply")
    arrow_vertices = arrow_vertices * (0.01 * voxel_vertices.max())

    ########################################################
    ## Generate a PlyFile object and initialize it with our points from the voxelgrid
    plyfile_vg = PlyFile(voxel_vertices, color=[255, 0 , 0])

    plyfile_vg.draw_arrows_for_sceneflow(voxel_vertices, sf_vg_vertices, arrow_vertices, arrow_faces)

    plyfile_vg.write_ply("vg_" + dataset + "_" + cfg.into + ".ply")
    ########################################################

    ########################################################
    ########################################################
    # ## Matplot ##
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # x, y, z = voxel_vertices[:,0], voxel_vertices[:,1], voxel_vertices[:,2]
    # u, v, w = sf_vg_vertices[:,0], sf_vg_vertices[:,1], sf_vg_vertices[:,2]
    #
    # # for i in range(len(u)):
    # #     print(u[i], v[i], w[i])
    #
    # ax.quiver(x, y, z, u, v, w, length=0.5, normalize=False)
    #
    # plt.show()
    ########################################################
    ########################################################


