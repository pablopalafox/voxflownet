import torch
import torch.nn as nn
import torch.nn.init as init
#import torchvision.models as models
import torch.nn.functional as F
from config import Config as cfg

import time as time

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

# Fully Connected Network
class FCN(nn.Module):
    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk * t, -1))
        x = F.relu(self.bn(x))
        return x.view(kk, t, -1)

# Voxel Feature Encoding layer
class VFE(nn.Module):
    def __init__(self, cin, cout):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin, self.units)

    def forward(self, x, mask):
        ## point-wise feauture
        pwf = self.fcn(x) # i.e.: ([5456, 35, 16])
        # print("pwf", pwf.shape)

        ## locally aggregated feature (max feature in voxel)
        laf = torch.max(pwf, 1)[0].unsqueeze(1).repeat(1, cfg.T, 1) # i.e.: ([5456, 35, 16])
        # print("laf", laf.shape)

        ## point-wise concat feature
        pwcf = torch.cat((pwf, laf), dim=2) # i.e.: ([5456, 35, 32])
        # print("pwcf", pwcf.shape)

        ## apply mask
        # print("mask prev", mask.shape)
        # print(mask)
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        # print(mask)
        # print("mask post", mask.shape)

        pwcf = pwcf * mask.float()
        # print("pwcf", pwcf.shape)

        return pwcf


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):
    def __init__(self, use_normals):
        super(SVFE, self).__init__()
        cin = 9 if use_normals else 6
        self.vfe_1 = VFE(cin, 32)
        self.vfe_2 = VFE(32, 128)
        self.fcn = FCN(128, 128)

    def forward(self, x):
        mask = torch.ne(torch.max(x, 2)[0], 0)

        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)

        ###################################################
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        """###################################################"""
        # element-wise max pooling
        x = torch.max(x, 1)[0]
        # print("max pool", x.shape)
        """###################################################"""
        # end.record()
        # torch.cuda.synchronize()
        # print("timing max: ", start.elapsed_time(end))
        ###################################################

        return x


# Convolutional Middle Layer
class CML(nn.Module):
    def __init__(self):
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(in_channels=128, out_channels=64, k=3, s=1, p=1, batch_norm=True)
        self.conv3d_2 = Conv3d(in_channels=64, out_channels=64, k=3, s=1, p=1, batch_norm=True)
        self.conv3d_3 = Conv3d(in_channels=64, out_channels=64, k=3, s=1, p=1, batch_norm=True)

    def forward(self, x):
        x = self.conv3d_1(inference=False, relu=True, x=x)
        x = self.conv3d_2(inference=False, relu=True, x=x)
        x = self.conv3d_3(inference=False, relu=True, x=x)
        return x


class VoxelNet(nn.Module):
    def __init__(self, use_normals):
        super(VoxelNet, self).__init__()
        self.svfe = SVFE(use_normals)
        self.cml = CML()

    def voxel_indexing(self, split, sparse_features, coords):
        dense_feature = torch.cuda.FloatTensor(128, cfg.batch_sizes[split], cfg.n_voxels, cfg.n_voxels, cfg.n_voxels).fill_(0)
        dense_feature[:, coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = sparse_features.transpose(0, 1)
        return dense_feature.transpose(0, 1)

    def forward(self, split, voxel_features, voxel_coords):
        # feature learning network
        ###################################################
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        ###################################################
        vwfs = self.svfe(voxel_features)
        # print("vwfs", vwfs.shape)
        ###################################################
        # end.record()
        # torch.cuda.synchronize()
        # print("timing 1: ", start.elapsed_time(end))
        ###################################################

        ###################################################
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        ###################################################
        vwfs = self.voxel_indexing(split, vwfs, voxel_coords)
        ###################################################
        # end.record()
        # torch.cuda.synchronize()
        # print("timing 2: ", start.elapsed_time(end))
        ###################################################

        # convolutional middle network
        ###################################################
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        ###################################################
        cml_out = self.cml(vwfs)
        ###################################################
        # end.record()
        # torch.cuda.synchronize()
        # print("timing 3: ", start.elapsed_time(end))
        ###################################################

        return cml_out


class SiameseVoxelNet(nn.Module):

    ######################################################
    ################### Siamese Model ####################
    ######################################################

    def __init__(self, verbose, use_bn, use_dropout, use_normals):
        super(SiameseVoxelNet, self).__init__()
        self.verbose = verbose

        self.vn = VoxelNet(use_normals)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv_1x1 = Conv3d(in_channels=32, out_channels=32, k=1, s=1, p=0, batch_norm=False)
        if use_dropout:
            self.drop = nn.Dropout()
        else:
            self.drop = None

        self.conv_concat_11 = Conv3d(in_channels=128, out_channels=64, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_concat_12 = Conv3d(in_channels=64, out_channels=64, k=3, s=1, p=1, batch_norm=use_bn)

        self.upscore1 = ConvTrans3d(64, 32, k=3, s=2, p=1, out_p=1, batch_norm=use_bn)
        self.upscore2 = ConvTrans3d(32, 3, k=3, s=2, p=1, out_p=1, batch_norm=False)

    def forward(self, inference, split, *input):

        ###################################################
        # start_forward = torch.cuda.Event(enable_timing=True)
        # end_forward = torch.cuda.Event(enable_timing=True)
        # start_forward.record()
        ###################################################

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
            x = self.vn(split, voxel_features, voxel_coords)
            ###################################################
            ###################################################
            # end.record()
            # torch.cuda.synchronize()
            # print("timing voxelNet: ", start.elapsed_time(end))
            # ###################################################

            if i == 0:
                pcl_t0 = torch.cat((pcl_t0, x), 0)
            elif i == 1:
                pcl_t1 = torch.cat((pcl_t1, x), 0)

        ###################################################
        # start_else = torch.cuda.Event(enable_timing=True)
        # end_else = torch.cuda.Event(enable_timing=True)
        # start_else.record()
        ###################################################

        ##########################
        ## CONCATENATE ENCODERS ##
        ##########################
        if self.verbose:
            print("pcl_t0", pcl_t0.shape)
            print("pcl_t1", pcl_t1.shape)

        x = torch.cat((pcl_t0, pcl_t1), 1)
        if self.verbose:
            print("concatenated ", x.shape)

        ################################
        ######### TOGETHER #############
        ################################
        # Conv Together 1
        x = self.conv_concat_11(inference=inference, relu=True, x=x)
        x = self.pool(x)
        x = self.conv_concat_12(inference=inference, relu=True, x=x)
        x = self.pool(x)

        # x = self.conv_concat_21(inference=inference, relu=True, x=x)
        # x = self.conv_concat_22(inference=inference, relu=True, x=x)
        # x = self.pool(x)

        if self.verbose:
            print("Conv Together:      ", x.shape)

        ##########################
        ######### DECODER ########
        ##########################
        # UpConv 1
        x = self.upscore1(inference=inference, relu=False, x=x)
        if self.verbose:
            print("upConv 1 ", x.shape)

        ## UpConv 2
        x = self.upscore2(inference=inference, relu=False, x=x)
        if self.verbose:
            print("upConv 2 ", x.shape)

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

        ###################################################
        # end_else.record()
        # torch.cuda.synchronize()
        # print("timing else: ", start_else.elapsed_time(end_else))
        ###################################################

        ##################################################
        # end_forward.record()
        # torch.cuda.synchronize()
        # print("timing forward: ", start_forward.elapsed_time(end_forward))
        ##################################################


        return sf_pred


class SiameseModel3D_1M_no_second_1x1conv(nn.Module):

    ######################################################
    ################### Siamese Model ####################
    ######################################################

    def __init__(self, verbose, use_bn, use_dropout):
        super(SiameseModel3D_1M_no_second_1x1conv, self).__init__()

        self.verbose = verbose

        ######################################################
        ## We use same padding all along                    ##
        ## For SAME Padding                                 ##
        ## p = (F - 1) / 2 , where F is the filter size     ##
        ## With F=3, this means a padding of p=1            ##
        ######################################################

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv_11 = Conv3d(in_channels=1, out_channels=8, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_12 = Conv3d(in_channels=8, out_channels=8, k=3, s=1, p=1, batch_norm=use_bn)

        self.conv_21 = Conv3d(in_channels=8, out_channels=16, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_22 = Conv3d(in_channels=16, out_channels=16, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_23 = Conv3d(in_channels=16, out_channels=16, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_24 = Conv3d(in_channels=16, out_channels=16, k=3, s=1, p=1, batch_norm=use_bn)

        self.conv_31 = Conv3d(in_channels=16, out_channels=32, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_32 = Conv3d(in_channels=32, out_channels=32, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_33 = Conv3d(in_channels=32, out_channels=32, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_34 = Conv3d(in_channels=32, out_channels=32, k=3, s=1, p=1, batch_norm=use_bn)

        self.conv_1x1 = Conv3d(in_channels=32, out_channels=32, k=1, s=1, p=0, batch_norm=False)
        if use_dropout:
            self.drop = nn.Dropout()
        else:
            self.drop = None

        self.conv_concat_11 = Conv3d(in_channels=64, out_channels=128, k=3, s=1, p=1, batch_norm=use_bn)
        self.conv_concat_12 = Conv3d(in_channels=128, out_channels=256, k=3, s=1, p=1, batch_norm=use_bn)

        self.upscore1 = ConvTrans3d(256, 32, k=3, s=2, p=1, out_p=1, batch_norm=use_bn)
        self.upscore2 = ConvTrans3d(32, 16, k=3, s=2, p=1, out_p=1, batch_norm=False)
        self.upscore3 = ConvTrans3d(16, 3, k=3, s=2, p=1, out_p=1, batch_norm=False)

    def voxel_indexing(self, split, coords):
        dense_feature = torch.cuda.FloatTensor(1, cfg.batch_sizes[split], cfg.n_voxels, cfg.n_voxels, cfg.n_voxels).fill_(0)
        dense_feature[:, coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = 1.0
        return dense_feature.transpose(0, 1)

    def forward(self, inference, split, *input):

        vg_t0 = torch.FloatTensor()
        vg_t1 = torch.FloatTensor()

        ## ENCODER ##
        # loop over the input voxelgrids (vg_t0 and vg_t1)
        for i in range(len(input)):
            x = input[i]
            x = self.voxel_indexing(split, x)

            if self.verbose:
                print("Initial shape: ", x.shape)
                N, Cin, D, H, W,  = x.shape
                input_size = D * H * W * Cin

            # Conv 1
            x = self.conv_11(inference=inference, relu=True, x=x)
            x = self.conv_12(inference=inference, relu=True, x=x)
            x = self.pool(x)
            if self.verbose:
                print("Conv 1:      ", x.shape)
                print("MaxPool 1:    ", x.shape)

            # Conv 2
            x = self.conv_21(inference=inference, relu=True, x=x)
            x = self.conv_22(inference=inference, relu=True, x=x)
            x = self.conv_23(inference=inference, relu=True, x=x)
            x = self.conv_24(inference=inference, relu=True, x=x)
            x = self.pool(x)
            if self.verbose:
                print("Conv 2:      ", x.shape)
                print("MaxPool 2:    ", x.shape)

            # Conv 3
            x = self.conv_31(inference=inference, relu=True, x=x)
            x = self.conv_32(inference=inference, relu=True, x=x)
            x = self.conv_33(inference=inference, relu=True, x=x)
            x = self.conv_34(inference=inference, relu=True, x=x)
            x = self.pool(x)
            if self.verbose:
                print("Conv 3:      ", x.shape)
                print("MaxPool 3:    ", x.shape)

            # Conv 1x1
            x = self.conv_1x1(inference=inference, relu=True, x=x)
            if self.drop is not None:
                x = self.drop(x)
            if self.verbose:
                print("conv3d_1x1: ", x.shape)

            if i == 0:
                vg_t0 = torch.cat((vg_t0.cuda(), x), 0)
            elif i == 1:
                vg_t1 = torch.cat((vg_t1.cuda(), x), 0)

        if self.verbose:
            N_, Cout_, D_, H_, W_ = vg_t0.shape
            output_size = Cout_ * D_ * H_ * W_
            print(input_size, ":", Cin, D, H, W)
            print(output_size, ":", Cout_, D_, H_, W_)
            print(input_size / output_size)


        ##########################
        ## CONCATENATE ENCODERS ##
        ##########################
        if self.verbose:
            print("vg_t0", vg_t0.shape)
            print("vg_t1", vg_t1.shape)

        x = torch.cat((vg_t0, vg_t1), 1)
        if self.verbose:
            print("concatenated ", x.shape)


        ################################
        ######### TOGETHER #############
        ################################
        # Conv Together 1
        x = self.conv_concat_11(inference=inference, relu=True, x=x)
        x = self.conv_concat_12(inference=inference, relu=True, x=x)
        if self.verbose:
            print("Conv Together:      ", x.shape)


        ##########################
        ######### DECODER ########
        ##########################
        # UpConv 1
        x = self.upscore1(inference=inference, relu=False, x=x)
        if self.verbose:
            print("upConv 1 ", x.shape)

        ## UpConv 2
        x = self.upscore2(inference=inference, relu=False, x=x)
        if self.verbose:
            print("upConv 2 ", x.shape)

        ## UpConv 3
        x = self.upscore3(inference=inference, relu=False, x=x)
        if self.verbose:
            print("upConv 3 ", x.shape)


        #sf_pred = x.permute(0, 2, 3, 4, 1)

        ## Convert to dense tensor
        # b = torch.LongTensor(input[0])
        # n_chunks = b.shape[0]
        # b_t = b.t()
        # chunks = b_t.chunk(chunks=n_chunks, dim=0)
        # sf_pred = sf_pred[chunks]
        # sf_pred = sf_pred.squeeze()

        #############################
        ## Convert to dense tensor ##
        #############################
        sf_pred = x.permute(0, 2, 3, 4, 1)
        voxel_coords_t0 = input[0]
        b = voxel_coords_t0
        n_chunks = b.shape[0]
        b_t = b.t()
        chunks = b_t.chunk(chunks=n_chunks, dim=0)
        sf_pred = sf_pred[chunks]
        sf_pred = sf_pred.squeeze()

        return sf_pred
