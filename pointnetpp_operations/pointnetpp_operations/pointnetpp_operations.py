
import os
import torch
from torch.autograd import Function
import torch.nn as nn
from typing import *

from torch.utils.cpp_extension import load
ppp_ops = load(name="ppp_ops",
               sources=[f"{os.path.dirname(os.path.abspath(__file__))}/pointnetpp_operations.cpp",
                        f"{os.path.dirname(os.path.abspath(__file__))}/pointnetpp_operations.cu"])


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest minimum distance
        :param ctx:
        :param xyz: (B, N, 3) tensor where N > npoint
        :param npoint: number of features in the sampled set
        :return: (B, npoint) tensor containing the set
        """
        assert(xyz.is_cuda)
        return ppp_ops.furthest_point_sampling_cuda(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """

        :param ctx:
        :param features: (B, C, N) tensor
        :param idx: (B, npoint) tensor of the features to gather
        :return: (B, C, npoint) tensor
        """
        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        assert (features.is_cuda and idx.is_cuda)
        return ppp_ops.gather_points_cuda(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = ppp_ops.group_points_grad_cuda(grad_out.contiguous(), idx, N)
        return grad_features, None


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: torch.Tensor
        :param known: (B, m, 3) tensor of unknown features
        :return: (B, n, 3) l2 distance to the three nearest neighbors; (B, n, 3) index of 3 nearest neighbors
        """

        assert(unknown.is_cuda and known.is_cuda)
        dist2, idx = ppp_ops.three_nn_cuda(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, c, m) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return: (B, c, n) tensor of the interpolated features
        """

        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        assert(features.is_cuda and idx.is_cuda and weight.is_cuda)
        return ppp_ops.three_interpolate_cuda(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param ctx:
        :param grad_out: (B, c, n) tensor with gradients of ouputs
        :return: (B, c, m) tensor with gradients of features
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = ppp_ops.three_interpolate_grad_cuda(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """

        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indices of features to group with
        :return: (B, C, npoint, nsample) tensor
        """

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        assert(features.is_cuda and idx.is_cuda)
        return ppp_ops.group_points_cuda(features, idx)

    @staticmethod
    def backward(ctx, grad_out: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return: (B, C, N) gradient of the features
        """

        idx, N = ctx.for_backwards

        grad_features = ppp_ops.group_points_grad_cuda(grad_out.contiguous(), idx, N)

        return grad_features, None


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """

        :param ctx:
        :param radius: radius of the balls
        :param nsample: maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return: (B, npoint, nsample) tensor with the indices of the features that form the query balls
        """
        assert(new_xyz.is_cuda and xyz.is_cuda)
        return ppp_ops.ball_query_cuda(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int):
        """
        Groups with a ball query of radius
        :param radius: Radius of ball
        :param nsample: Maximum number of features to gather in the ball
        """
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample = radius, nsample

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """

        :param xyz: xyz coordinates of the features (B, N, 3)
        :param new_xyz: centroids (B, npoint, 3)
        :param features: Descriptors of the features (B, N, C)
        :return: (B, 3 + C, npoint, nsample) tensor
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = grouping_operation(xyz.transpose(1, 2).contiguous(), idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features.transpose(1, 2).contiguous(), idx) # (B, C, npoint, nsample)
            new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
        else:
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    def __init__(self):
        """
        Groups all features
        """
        super(GroupAll, self).__init__()

    def forward(self, xyz, new_xyz: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """

        :param xyz: xyz coordinates of the features (B, N, 3)
        :param new_xyz: Ignored
        :param features: Descriptors of the features (B, N, C)
        :return: (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            features = features.transpose(1, 2).contiguous() # (B, C, N)
            grouped_features = features.unsqueeze(2)
            new_features = torch.cat(
                [grouped_xyz, grouped_features], dim=1
            )  # (B, 3 + C, 1, N)
        else:
            new_features = grouped_xyz

        return new_features


ball_query = BallQuery.apply
furthest_point_sample = FurthestPointSampling.apply
gather_operation = GatherOperation.apply
three_nn = ThreeNN.apply
three_interpolate = ThreeInterpolate.apply
grouping_operation = GroupingOperation.apply
