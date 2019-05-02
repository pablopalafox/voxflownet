import torch
import torch.nn as nn
import torch.nn.functional as F

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class _Loss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce


##############################################################################


class EPELoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(EPELoss, self).__init__(size_average, reduce)
        print("EPELoss")

    def forward(self, pred, target):
        """
        input must be (N, D, H, W, Channels=3)
        :param input:
        :param target:
        :return:
        """
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        _assert_no_grad(target)

        loss = torch.sum((pred - target)**2, dim=-1)
        loss = torch.sqrt(loss)  # For every voxel, square root of sum of components
        return torch.mean(loss) if self.size_average else torch.sum(loss)


class MyL1Loss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(MyL1Loss, self).__init__(size_average, reduce)

    def forward(self, pred, target):
        """
        input must be (N, D, H, W, Channels=3)
        :param input:
        :param target:
        :return:
        """
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        _assert_no_grad(target)
        loss = torch.abs(pred - target)
        if not self.reduce:
            return loss
        return torch.mean(loss) if self.size_average else torch.sum(loss)


class MyMSELoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(MyMSELoss, self).__init__(size_average, reduce)

    def forward(self, pred, target):
        """
        input must be (N, D, H, W, Channels=3)
        :param input:
        :param target:
        :return:
        """
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        _assert_no_grad(target)
        loss = pred - target
        loss = torch.pow(loss, 2.0)
        if not self.reduce:
            return loss
        return torch.mean(loss) if self.size_average else torch.sum(loss)


class MySmoothL1Loss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(MySmoothL1Loss, self).__init__(size_average, reduce)

    def forward(self, pred, target):
        """
        input must be (N, D, H, W, Channels=3)
        :param input:
        :param target:
        :return:
        """
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        _assert_no_grad(target)

        target_norm = torch.sqrt(torch.sum(target**2, dim=-1))
        mask_nonzero = (target_norm != 0.0).detach()
        num_nonzero = torch.nonzero(mask_nonzero).size(0)

        if num_nonzero == 0:
            print("ACHTUNG", num_nonzero)

        abs_diff = torch.sqrt(torch.sum((pred - target)**2, dim=-1))
        mask_smaller = (abs_diff < 1.0).detach()

        mask_smaller = mask_smaller & mask_nonzero
        mask_bigger = ~mask_smaller & mask_nonzero

        loss = abs_diff[mask_bigger].sum() - 0.5
        loss += 0.5 * (abs_diff[mask_smaller]**2).sum()
        return (loss / num_nonzero) if self.size_average else loss


class MySmoothL1LossSparse(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(MySmoothL1LossSparse, self).__init__(size_average, reduce)
        print("MySmoothL1LossSparse")


    def forward(self, pred, target):
        """
        :param pred:
        :param target:
        :return:
        """
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        _assert_no_grad(target)

        if target.size == 0 or pred.size == 0:
            raise Exception("Ups... empty pred or target vector!")

        abs_diff = torch.sum((pred - target)**2, dim=-1)
        abs_diff = torch.sqrt(abs_diff)
        mask_smaller = (abs_diff < 1.0).detach()
        mask_bigger = ~mask_smaller

        loss = (abs_diff[mask_bigger] - 0.5).sum()
        loss += 0.5 * (abs_diff[mask_smaller]**2).sum()
        return (loss / target.size(0)) if self.size_average else loss


class AngularErrorLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(AngularErrorLoss, self).__init__(size_average, reduce)

    def forward(self, pred, target):
        _assert_no_grad(target)

        #shit = torch.nonzero(target.data)[0:2]

        # Numerator
        num = torch.mul(pred, target)
        num = torch.sum(num, dim=4)
        num = torch.add(num, 1)

        # Denominator
        pred = torch.pow(pred, 2.0)
        pred = torch.sum(pred, dim=4)
        pred = torch.add(pred, 1)

        target = torch.pow(target, 2.0)
        target = torch.sum(target, dim=4)
        target = torch.add(target, 1)

        denom = torch.mul(pred, target)
        denom = torch.sqrt(denom)

        loss = torch.div(num, denom)
        loss = 1.0 / loss
        print(loss)
        print(loss)
        return torch.mean(loss) if self.size_average else torch.sum(loss)


class BerHuLoss(_Loss):

    def __init__(self, size_average=True, reduce=True):
        super(BerHuLoss, self).__init__(size_average, reduce)

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        abs_diff = (pred - target).abs()

        c = 0.2 * torch.max(abs_diff).item()
        mask = (abs_diff < c).detach()

        loss = abs_diff[mask].sum()
        loss += (torch.pow(abs_diff[~mask], 2) / (2.0 * c) + c / 2.0).sum()
        return loss / abs_diff.numel() if self.size_average else loss


class VoxelLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(size_average=False)
        self.alpha = alpha
        self.beta = beta

    def forward(self, rm, psm, pos_equal_one, neg_equal_one, targets):

        p_pos = F.sigmoid(psm.permute(0,2,3,1))
        rm = rm.permute(0,2,3,1).contiguous()
        rm = rm.view(rm.size(0),rm.size(1),rm.size(2),-1,7)
        targets = targets.view(targets.size(0),targets.size(1),targets.size(2),-1,7)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).expand(-1,-1,-1,-1,7)

        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg

        cls_pos_loss = -pos_equal_one * torch.log(p_pos + 1e-6)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)

        cls_neg_loss = -neg_equal_one * torch.log(1 - p_pos + 1e-6)
        cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + 1e-6)

        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss
        return conf_loss, reg_loss



