import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)
import os

path_to_sceneflow = "/mnt/raid/pablo/data/FlyingThings3D/32/sceneflow"
path_to_voxelgrid = "/mnt/raid/pablo/data/FlyingThings3D/32/voxelgrid"

# bad = ['TRAIN-C-0098-0009', 'TRAIN-C-0169-0007', 'TRAIN-C-0502-0012', 'TRAIN-C-0092-0011', 'TRAIN-A-0719-0012', 'TRAIN-A-0313-0007', 'TRAIN-B-0459-0012', 'TRAIN-C-0035-0014', 'TRAIN-C-0020-0010', 'TRAIN-A-0325-0009', 'TRAIN-A-0434-0013', 'TRAIN-C-0644-0012', 'TRAIN-A-0001-0014', 'TRAIN-C-0054-0013', 'TRAIN-C-0001-0008', 'TRAIN-A-0084-0006', 'TRAIN-C-0591-0011', 'TRAIN-A-0230-0013', 'TRAIN-C-0359-0014', 'TRAIN-B-0289-0008', 'TRAIN-A-0568-0014', 'TRAIN-A-0587-0009', 'TRAIN-C-0475-0007', 'TRAIN-A-0372-0014', 'TRAIN-B-0657-0009', 'TRAIN-B-0691-0009', 'TRAIN-C-0388-0010', 'TRAIN-B-0267-0006', 'TRAIN-B-0618-0008', 'TRAIN-B-0293-0009', 'TRAIN-C-0140-0006', 'TRAIN-A-0582-0014']
# bad2 = []
# for b in bad:
#     b = b.split('-')
#     b = b[1:3]
#     b = '/'.join(b)
#     print(b)
#     bad2.append(b)
#     b = b + ".npy"
#     # sf = os.path.join(path_to_sceneflow, b)
#     # vg = os.path.join(path_to_voxelgrid, b)
#     # sceneflow = np.load(sf)
#     # voxelgrid = np.load(vg)
#     #
#     # nonzero_sf = np.count_nonzero(sceneflow)
#     # nonzero_vg = np.count_nonzero(sceneflow)
#     # if nonzero_sf == 0:
#     #     print("sf", b)
#     # if nonzero_vg == 0:
#     #     print("vg", b)
#
# print(bad2)
#
# exit()


data_splits = ["TEST", "TRAIN"]
letters = ["A", "B", "C"]

count = 0
bad_samples = []
for data_split in data_splits:
    for letter in letters:
        for number in os.listdir(os.path.join(path_to_sceneflow, data_split, letter)):
            path_to_sequence = os.path.join(path_to_sceneflow, data_split, letter, number)
            for sample in os.listdir(os.path.join(path_to_sequence)):
                path_to_sample = os.path.join(path_to_sequence, sample)
                sceneflow = np.load(path_to_sample)

                if (sceneflow.size == 0):
                    thingy = os.path.join(data_split, letter, number, sample)
                    print("In", thingy, "no points")
                    bad_samples.append(thingy)

                # ## Count non zero
                # n_of_nonzero = np.count_nonzero(sceneflow)
                # if n_of_nonzero == 0:
                #     thingy = os.path.join(data_split, letter, number, sample)
                #     print("In", thingy, "no points")
                #     empty_samples.append(thingy)

                # Count non-Nan
                n_of_nonNan = np.count_nonzero(~np.isnan(sceneflow))
                if sceneflow.size - n_of_nonNan != 0:
                    thingy = os.path.join(data_split, letter, number, sample)
                    print("In", thingy, "NaN points")
                    bad_samples.append(thingy)
                    count += 1
#
# empty_samples.sort()
#
# with open("nan_sceneflow.txt", "w") as f:
#     for empty_sample in empty_samples:
#         f.write(empty_sample + "\n")
#
# print("Count", count)
#

#################################################################

# with open("/home/pablo/sceneflow/nan_sceneflow.txt", "r") as f:
#     zero_sfs = f.readlines()
#
# # print(zero_sfs)
#
# for zero_sf in zero_sfs:
#     sf_path = os.path.join(path_to_sceneflow, zero_sf.rstrip())
#     print(sf_path)
#     os.remove(sf_path)
#
#     sf_path_dir = sf_path.split('/')
#     sf_path_dir = sf_path_dir[1:-1]
#     sf_path_dir = "/" + '/'.join(sf_path_dir)
#
#     if not os.listdir(sf_path_dir):
#         os.rmdir(sf_path_dir)
