
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Tuple
from pointnetpp_operations import pointnetpp_operations
from config import Config as cfg

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x=self.bn(x)
        if self.activation:
            return F.relu(x,inplace=True)
        else:
            return x


class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(out_channels) if batch_norm else None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm3d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)

    def forward(self, inference, relu, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn.train(False if inference else True).cuda()(x)
        return F.relu(x, inplace=True) if relu else x


class ConvTrans3d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, out_p=0, batch_norm=True, relu=True):
        super(ConvTrans3d, self).__init__()
        self.convTrans = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, output_padding=out_p)
        self.bn = nn.BatchNorm3d(out_channels) if batch_norm else None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm3d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)

    def forward(self, inference, relu, x):
        x = self.convTrans(x)
        if self.bn is not None:
            x = self.bn.train(False if inference else True).cuda()(x)
        return F.relu(x, inplace=True) if relu else x


def build_mlp(channels):
    assert(len(channels) == 4)
    return nn.Sequential(
        nn.Conv2d(channels[0], channels[1], 1),
        nn.BatchNorm2d(channels[1]),
        nn.ReLU(),
        nn.Conv2d(channels[1], channels[2], 1),
        nn.BatchNorm2d(channels[2]),
        nn.ReLU(),
        nn.Conv2d(channels[2], channels[3], 1),
        nn.BatchNorm2d(channels[3]),
        nn.ReLU()
    )


class QueryAndGroup(nn.Module):
    def __init__(self, npoints, radius, nsample):
        """
        Grouping points for each voxel
        :param subsampler: Sub-sampling rate (integer by which we will then divide the number of pts)
        :param npoints: Number of "balls" (super-points)
        :param radius: radius to group with
        :param nsamples: Number of samples in each "ball"
        """
        super(QueryAndGroup, self).__init__()

        self.npoints = npoints
        self.grouper = pointnetpp_operations.QueryAndGroup(radius, nsample)

    def forward(self, xyz_features) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward
        :param xyz: (B, N, 3) tensor of the N xyz coordinates of the features
        :param normals: (B, N, C) tensor of the descriptors of the the features
        :param xyz_voxel: (B, n, 3) tensor of the n xyz coordinates of the features inside a voxel
        :param normals_voxel: (B, n, 3) tensor of the n normals inside a voxel
        :return: (B, npoint, 3) tensor of the new features' xyz;
        (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        xyz_all = xyz_features[0]
        normals = xyz_features[1]
        xyz_voxel = xyz_features[2]
        normals_voxel = xyz_features[3]


        tmp = pointnetpp_operations.furthest_point_sample(xyz_voxel, self.npoints)

        xyz_flipped = xyz_voxel.transpose(1, 2).contiguous()
        new_xyz = (
            pointnetpp_operations.gather_operation(
                xyz_flipped, tmp
            ).transpose(1, 2).contiguous()
        )

        normals_flipped = normals_voxel.transpose(1, 2).contiguous()
        new_normals = (
            pointnetpp_operations.gather_operation(
                normals_flipped, tmp
            ).transpose(1, 2).contiguous()
        )

        new_features = self.grouper(xyz_all, new_xyz)  # (B, C, npoint, nsample)
        new_features_normals = self.grouper(xyz_all, new_xyz, normals)  # (B, C, npoint, nsample)

        return new_xyz, new_normals, new_features, new_features_normals


class PointNetMLP(nn.Module):
    def __init__(self, mlp_spec):
        """
        Apply Multi Layer Perceptron to a set of pre-computed voxel features
        :param
        :param
        :param
        """
        super(PointNetMLP, self).__init__()

        self.mlp = build_mlp(mlp_spec)

    def forward(self, voxels_features):
        """
        Forward
        :param voxels_features: (B, V, 3+C, npoints, nsamples)
        :return: (B, V, npoint, 3) tensor of the new features' xyz;
        (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features = self.mlp(voxels_features)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        new_features = new_features.transpose(2, 1).contiguous()
        new_features = torch.max(new_features, 1)[0]

        return new_features


class PointnetSAModuleMSG(nn.Module):
    def __init__(self, subsampler, radii, nsamples, mlps):
        """
        Pointnet Set Abstraction Layer with multiscale grouping
        :param subsampler: Sub-sampling rate (integer by which we will then divide the number of pts)
        :param radii: list of radii to group with
        :param nsamples: Number of samples in each ball query
        :param mlps: Spec of the pointnet before the global max_pool for each scale
        """
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.subsampler = subsampler
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnetpp_operations.QueryAndGroup(radius, nsample)
                if subsampler is not None
                else pointnetpp_operations.GroupAll()
            )
            mlp_spec = mlps[i]
            mlp_spec[0] += 3

            self.mlps.append(build_mlp(mlp_spec))

    def forward(self, xyz_features) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :return: (B, npoint, 3) tensor of the new features' xyz;
        (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        xyz = xyz_features[0] if len(xyz_features) > 1 and isinstance(xyz_features, tuple) else xyz_features
        features = xyz_features[1] if len(xyz_features) > 1 and isinstance(xyz_features, tuple) else None

        npoint = xyz.shape[1] // self.subsampler if self.subsampler is not None else None

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnetpp_operations.gather_operation(
                xyz_flipped, pointnetpp_operations.furthest_point_sample(xyz, npoint)
            ).transpose(1, 2).contiguous()
            if npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                    xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features = new_features.transpose(2, 1).contiguous()
            new_features = torch.max(new_features, 1)[0]

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModule(PointnetSAModuleMSG):
    def __init__(self, mlp, subsampler=None, radius=None, nsample=None):
        """
        Pointnet Set Abstraction Layer
        :param mlp: Spec of the pointnet before the global max_pool
        :param subsampler: Sub-sampling rate (integer by which we will then divide the number
        :param radius: Radius of ball
        :param nsample: Number of samples in the ball query
        """
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            subsampler=subsampler,
            radii=[radius],
            nsamples=[nsample]
        )


class PointNet(nn.Module):
    """
    Use pointnet module to compute a feature for a set of points (points inside every voxel)
    """
    def __init__(self, cin, nfeatures):
        super(PointNet, self).__init__()
        input_channels = cin
        self.nfeatures = nfeatures
        self.mlp = PointNetMLP(mlp_spec=[input_channels, cfg.nfeat//2, cfg.nfeat//2, cfg.nfeat])

    def voxel_indexing(self, split, sparse_features, coords):
        dense_feature = torch.cuda.FloatTensor(self.nfeatures, cfg.batch_sizes[split],
                                               cfg.n_voxels, cfg.n_voxels, cfg.n_voxels).fill_(0)
        dense_feature[:, coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = sparse_features.transpose(0, 1)
        return dense_feature.transpose(0, 1)

    def forward(self, split, voxel_features, voxel_coords):
        new_voxel_features = self.mlp(voxel_features)
        new_voxel_features = self.voxel_indexing(split, new_voxel_features, voxel_coords)
        return new_voxel_features


class SiamesePointNet(nn.Module):
    def __init__(self, verbose, use_bn, use_dropout, use_local_features, use_normals, nfeat):
        super(SiamesePointNet, self).__init__()
        self.verbose = verbose

        print("use_local_features =", use_local_features, "use_normals =", use_normals)

        print(use_local_features)
        if use_local_features:
            cin = 6 if use_normals else 3
        else:
            print("hi")
            cin = 9 if use_normals else 6

        print("cin", cin)

        self.pn = PointNet(cin=cin, nfeatures=nfeat)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.drop = nn.Dropout() if use_dropout else None

        self.conv_concat_11 = Conv3d(in_channels=nfeat*2, out_channels=nfeat*4, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_concat_12 = Conv3d(in_channels=nfeat*4, out_channels=nfeat*8, k=3, s=1, p=1, batch_norm=use_bn)

        self.conv_concat_21 = Conv3d(in_channels=nfeat*8, out_channels=nfeat*16, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_concat_22 = Conv3d(in_channels=nfeat*16, out_channels=nfeat*16, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_concat_23 = Conv3d(in_channels=nfeat*16, out_channels=nfeat*16, k=3, s=1, p=1, batch_norm=use_bn)

        self.upscore1 = ConvTrans3d(nfeat*16, nfeat*8, k=3, s=2, p=1, out_p=1, batch_norm=use_bn)
        self.upscore2 = ConvTrans3d(nfeat*8, nfeat, k=3, s=2, p=1, out_p=1, batch_norm=use_bn)
        self.upscore3 = ConvTrans3d(nfeat, 3, k=3, s=2, p=1, out_p=1, batch_norm=False)

    def forward(self, inference, split, *input):
        pcl_t0 = torch.cuda.FloatTensor()
        pcl_t1 = torch.cuda.FloatTensor()

        ## ENCODER ##
        # loop over the input pointclouds (pcl_data_t0 and pcl_data_t1)
        for i in range(2):
            voxel_features = input[i][0]
            voxel_coords = input[i][1]

            ###################################################
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            ###################################################
            ###################################################
            #print(voxel_features.shape)
            x = self.pn(split, voxel_features, voxel_coords)
            # print(x.shape)
            #exit()
            ###################################################
            ###################################################
            # end.record()
            # torch.cuda.synchronize()
            # print("timing pointNet: ", start.elapsed_time(end))
            ###################################################

            if i == 0:
                pcl_t0 = torch.cat((pcl_t0, x), 0)
            elif i == 1:
                pcl_t1 = torch.cat((pcl_t1, x), 0)

        x = torch.cat((pcl_t0, pcl_t1), 1)
        
        ################################
        ######### TOGETHER #############
        ################################
        # Conv Together 1
        x = self.conv_concat_11(inference=inference, relu=True, x=x)
        if self.verbose:
            print("conv together 1", x.shape)
        x = self.pool(x)
        if self.verbose:
            print("pool 1", x.shape)
        x = self.conv_concat_12(inference=inference, relu=True, x=x)
        if self.verbose:
            print("conv together 2", x.shape)
        x = self.pool(x)
        if self.verbose:
            print("pool 2:      ", x.shape)
        x = self.conv_concat_21(inference=inference, relu=True, x=x)
        x = self.pool(x)
        if self.verbose:
            print("Conv Together:      ", x.shape)
        x = self.conv_concat_22(inference=inference, relu=True, x=x)
        if self.verbose:
            print("Conv Together:      ", x.shape)
        x = self.conv_concat_23(inference=inference, relu=True, x=x)
        if self.verbose:
            print("Conv Together:      ", x.shape)

        if self.verbose:
            print("Conv Together:      ", x.shape)

        ##########################
        ######### DECODER ########
        ##########################
        # UpConv 1
        x = self.upscore1(inference=inference, relu=False, x=x)
        if self.verbose:
            print("upConv 1 ", x.shape)

        # UpConv 2
        x = self.upscore2(inference=inference, relu=False, x=x)
        if self.verbose:
            print("upConv 2 ", x.shape)

        ## UpConv 2
        x = self.upscore3(inference=inference, relu=False, x=x)
        if self.verbose:
            print("upConv 3 ", x.shape)


        #############################
        ## Convert to dense tensor ##
        #############################
        sf_pred = x.permute(0, 2, 3, 4, 1)
        voxel_coords_t0 = input[0][1]
        b = voxel_coords_t0
        n_chunks = b.shape[0]
        b_t = b.t()
        chunks = b_t.chunk(chunks=n_chunks, dim=0)
        sf_pred = sf_pred[chunks]
        sf_pred = sf_pred.squeeze()

        return sf_pred


class SiamesePointNetSkipConnections(nn.Module):
    def __init__(self, verbose, use_bn, use_dropout, use_local_features, use_normals, nfeat):
        super(SiamesePointNetSkipConnections, self).__init__()
        self.verbose = verbose

        if use_local_features:
            cin = 6 if use_normals else 3
        else:
            cin = 9 if use_normals else 6

        self.pn = PointNet(cin=cin, nfeatures=nfeat)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.drop = nn.Dropout() if use_dropout else None

        self.conv_concat_11 = Conv3d(in_channels=nfeat * 2, out_channels=nfeat * 4, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_concat_12 = Conv3d(in_channels=nfeat * 4, out_channels=nfeat * 8, k=3, s=1, p=1, batch_norm=use_bn)

        self.conv_concat_21 = Conv3d(in_channels=nfeat * 8, out_channels=nfeat * 16, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_concat_22 = Conv3d(in_channels=nfeat * 16, out_channels=nfeat * 16, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_concat_23 = Conv3d(in_channels=nfeat * 16, out_channels=nfeat * 16, k=3, s=1, p=1, batch_norm=use_bn)

        self.upscore1 = ConvTrans3d(nfeat * 16, nfeat * 8, k=3, s=2, p=1, out_p=1, batch_norm=use_bn)
        self.upscore2 = ConvTrans3d(nfeat * 8, nfeat, k=3, s=2, p=1, out_p=1, batch_norm=use_bn)
        self.upscore3 = ConvTrans3d(nfeat, 3, k=3, s=2, p=1, out_p=1, batch_norm=False)

    def forward(self, inference, split, *input):
        pcl_t0 = torch.cuda.FloatTensor()
        pcl_t1 = torch.cuda.FloatTensor()

        ## ENCODER ##
        # loop over the input pointclouds (pcl_data_t0 and pcl_data_t1)
        for i in range(2):
            voxel_features = input[i][0]
            voxel_coords = input[i][1]

            ###################################################
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            ###################################################
            ###################################################
            x = self.pn(split, voxel_features, voxel_coords)
            ###################################################
            ###################################################
            # end.record()
            # torch.cuda.synchronize()
            # print("timing pointNet: ", start.elapsed_time(end))
            ###################################################

            print(x.shape)

            if i == 0:
                pcl_t0 = torch.cat((pcl_t0, x), 0)
            elif i == 1:
                pcl_t1 = torch.cat((pcl_t1, x), 0)

        x = torch.cat((pcl_t0, pcl_t1), 1)

        ################################
        ######### TOGETHER #############
        ################################
        # Conv Together 1
        x = self.conv_concat_11(inference=inference, relu=True, x=x)
        if self.verbose:
            print("conv 11", x.shape)
        x = self.pool(x)
        if self.verbose:
            print("pool 11", x.shape)
        x = self.conv_concat_12(inference=inference, relu=True, x=x)
        if self.verbose:
            print("conv 12", x.shape)
        x = self.pool(x)
        if self.verbose:
            print("pool 12:      ", x.shape)

        x = self.conv_concat_21(inference=inference, relu=True, x=x)
        x = self.pool(x)
        if self.verbose:
            print("Conv + Pool 21:      ", x.shape)

        x = self.conv_concat_22(inference=inference, relu=True, x=x)
        if self.verbose:
            print("Conv 22:      ", x.shape)

        x = self.conv_concat_23(inference=inference, relu=True, x=x)
        if self.verbose:
            print("Conv 23:      ", x.shape)


        ## 1x1 conv



        ##########################
        ######### DECODER ########
        ##########################
        # UpConv 1
        x = self.upscore1(inference=inference, relu=False, x=x)
        if self.verbose:
            print("upConv 1 ", x.shape)

        # UpConv 2
        x = self.upscore2(inference=inference, relu=False, x=x)
        if self.verbose:
            print("upConv 2 ", x.shape)

        ## UpConv 2
        x = self.upscore3(inference=inference, relu=False, x=x)
        if self.verbose:
            print("upConv 3 ", x.shape)

        exit()

        #############################
        ## Convert to dense tensor ##
        #############################
        sf_pred = x.permute(0, 2, 3, 4, 1)
        voxel_coords_t0 = input[0][1]
        b = voxel_coords_t0
        n_chunks = b.shape[0]
        b_t = b.t()
        chunks = b_t.chunk(chunks=n_chunks, dim=0)
        sf_pred = sf_pred[chunks]
        sf_pred = sf_pred.squeeze()

        return sf_pred