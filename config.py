import os

DATA_TYPES_3D = {
    'POINTCLOUD' : "pointcloud",
    'BOTH' : "pointcloud_voxelgrid"
}

class Config:
    ## Camera intrinsics and baseline
    fx = 1050  # in pixels
    fy = 1050  # in pixels
    cx = 479.5  # in pixels
    cy = 269.5  # in pixels
    baseline = 1.0

    ## Threshold to determine whether a matrix is a rotation matrix or not
    ## if (1 - det) > threshold: "It is not a rotation matrix"
    rot_matrix_threshold = 1e-01

    #######################################################################
    #######################################################################
    ## Number of voxels in every direction (x, y, z)
    n_voxels = 32

    ## Number of features pointnet
    nfeat = 64

    # maxiumum number of points per voxel
    T = 32

    # minimum number of points per voxel
    t = 5

    # maximum depth for point cloud
    max_z = 100

    # points cloud range for voxelization
    xrange = (-15, 15)
    yrange = (-10, 10)
    zrange = (5, 40)
    # xrange = (-2.5, 2.5)
    # yrange = (-2.5, 2.5)
    # zrange = (7.5, 12.5)

    vx = (xrange[1] - xrange[0]) / n_voxels
    vy = (yrange[1] - yrange[0]) / n_voxels
    vz = (zrange[1] - zrange[0]) / n_voxels

    #######################################################################
    #################### Params for optimization ##########################
    OVERFIT = False

    do_train = True
    do_validation = True
    do_eval = True

    epochs = 100
    use_dropout = False if OVERFIT else True
    use_batchnorm = True

    #####################################
    # use_local_features = False
    # use_normals = False
    #####################################

    batch_sizes = {
        "train" : 2,
        "val" : 2,
        "eval" : 1
    }

    data_type_3d = DATA_TYPES_3D['BOTH'] # Input data type
    model_name = "SiamesePointNet" # SiameseVoxelNet or
                                  # SiameseModel3D or
                                  # SiamesePointNet

    model_quality_to_use_at_eval = "BEST"
    model_dir_to_use_at_eval = "2.71913-1.45009_04-25-2019_21-26-34_EPELoss_localTrue_normalTrue_lr0.0001_lrDecayFalse_dropoutFalse_batchNormTrue_wDecay0_bs8_e2_data-A0000-0049"

    optimizer_type = 'adam'

    # loss_type = 'MySmoothL1LossSparse' # 'MSELoss', 'MySmoothL1LossSparse', 'L1Loss', 'EPELoss', 'BerHuLoss'
    size_average = True

    learning_rate = 1e-04
    use_lr_decay = False
    start_lr_decay = 20  # epochs
    lr_gamma = 0.7
    lr_decay_step_size = 20 # epochs

    # weight_decay = 2e-07

    val_every = 300  # in train batches
    batches_to_val = 40
    best_model_loss = float('Inf')
    best_model_epe = float('Inf')

    #######################################################################
    #######################################################################

    def __init__(self, system='Linux'):

        self.system = str(system)

        self.dataset = 'FlyingThings3D' # Options: 'FlyingThings3D' or 'Driving'
        self.into = 'into_future' # Options: 'into_future' or 'into_past'

        if self.system == 'Linux':
            ## Server
            self.dataset_path = "/mnt/raid/pablo/Datasets/" + self.dataset
        elif self.system == 'Windows':
            ## Windows local machine
            self.dataset_path = "E:/datasets/" + self.dataset

        ## Paths to dataset
        if self.system == 'Linux':
            ## Server
            self.root_dir = "/mnt/raid/pablo"
        elif self.system == 'Windows':
            ## Windows
            self.root_dir = "E:/"

        self.np_datasets_base_path = "data"
        self.data_dir = {
            'pointcloud' : os.path.join(self.root_dir, self.np_datasets_base_path, self.dataset,
                                        "pointcloud"),
            'pointcloud_voxelgrid' : os.path.join(self.root_dir,
                                                  self.np_datasets_base_path, self.dataset,
                                                  "pointcloud_voxelgrid_15_15_10_10_5_40")
        }

        self.sequences_to_train = ["A/0000-0049"] if self.OVERFIT else "ALL"
        self.sequences_to_train_str = self.get_str_from_list(self.sequences_to_train)
        self.sequences_to_eval = ["A/0000"] if self.OVERFIT else "ALL"
        self.sequences_to_draw = ["A-0000", "B-0000", "C-0000"]

        ## Path to experiments' folder
        experiments_base_path = "experiments"
        self.log_dir_base = {
            'pointcloud': os.path.join(self.root_dir, experiments_base_path, self.dataset,
                                       "pointcloud"),
            'pointcloud_voxelgrid': os.path.join(self.root_dir, experiments_base_path,
                                                 self.dataset, "pcl_" + str(self.n_voxels))
        }

        ## Data source for Tensorboard
        self.data_source = {
            'pointcloud': self.dataset,
            'pointcloud_voxelgrid': self.dataset + "_pcl_" + str(self.n_voxels),
        }

    def get_str_from_list(self, list):
        if list is None:
            return ""
        output_str = ''
        for element in list:
            output_str += element.replace('/', '')
        return output_str



