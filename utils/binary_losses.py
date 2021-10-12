import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, reduction='mean'):
        """Tversky loss for binary segmentation
        Args:
            alpha: penalty for false positives. Larger alpha weighs precision higher.
            beta: penalty for false negative. Larger beta weighs recall higher
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        Shapes:
            output: A tensor of shape [N, 1, (d,) h, w] without sigmoid activation function applied
            target: A tensor of shape same with output or shape of [N, (d,) h, w]
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        super(BinaryTverskyLoss, self).__init__()
        s = alpha + beta
        self.alpha = alpha / s
        self.beta = beta / s
        self.smooth = 1e-5
        self.reduction = reduction

    def forward(self, output, target, use_sigmoid=True):
        batch_size = output.size(0)

        if use_sigmoid:
            output = torch.sigmoid(output).view(batch_size, -1)
        else:
            output = output.view(batch_size, -1)
        target = target.view(batch_size, -1).float()

        P_G = torch.sum(output * target, 1)  # TP
        P_NG = torch.sum(output * (1 - target), 1)  # FP
        NP_G = torch.sum((1 - output) * target, 1)  # FN

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.smooth)

        loss = 1. - tversky_index

        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)

        return loss


class FocalBinaryTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3, reduction='mean'):
        super().__init__()
        s = alpha + beta
        self.alpha = alpha / s
        self.beta = beta / s
        self.gamma = gamma
        self.smooth = 1e-5
        self.reduction = reduction

    def forward(self, output, target, use_sigmoid=True):
        batch_size = output.size(0)

        if use_sigmoid:
            output = torch.sigmoid(output).view(batch_size, -1)
        else:
            output = output.view(batch_size, -1)
        target = target.view(batch_size, -1).float()

        P_G = torch.sum(output * target, 1)  # TP
        P_NG = torch.sum(output * (1 - target), 1)  # FP
        NP_G = torch.sum((1 - output) * target, 1)  # FN

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.smooth)

        tversky = 1. - tversky_index

        loss = tversky ** (1 / self.gamma)

        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    Shapes:
        output: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with output
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, reduction='mean', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1   # suggest set a large number when TP is large
        self.reduction = reduction
        self.batch_dice = False # treat a large map when True
        if 'batch_loss' in kwargs.keys():
            self.batch_dice = kwargs['batch_loss']

    def forward(self, output, target, use_sigmoid=True):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        if use_sigmoid:
            output = torch.sigmoid(output)

        dim0= output.shape[0]
        if self.batch_dice:
            dim0 = 1

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.pow(2) + target.pow(2), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class WBCEWithLogitLoss(nn.Module):
    """
    Weighted Binary Cross Entropy.
    `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1.
    To decrease the number of false positives, set β<1.
    Args:
            @param weight: positive sample weight
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, weight=1.0, reduction='mean'):
        super(WBCEWithLogitLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        weight = float(weight)
        self.weight = weight
        self.reduction = reduction
        self.smooth = 0.01

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        output = torch.sigmoid(output)
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)

        # loss = self.bce(output, target)
        loss = -self.weight * target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, output, target):
        output_flat = output.view(output.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(output_flat, target_flat.float(), reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(output_flat, target_flat.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class WBCE_DiceLoss(nn.Module):
    def __init__(self, alpha=1.0, weight=1.0, reduction='mean'):
        """
        combination of Weight Binary Cross Entropy and Binary Dice Loss
        Args:
            @param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            @param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
            @param alpha: weight between WBCE('Weight Binary Cross Entropy') and binary dice, apply on WBCE
        Shapes:
            output: A tensor of shape [N, *] without sigmoid activation function applied
            target: A tensor of shape same with output
        """
        super(WBCE_DiceLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        assert 0 <= alpha <= 1, '`alpha` should in [0,1]'
        self.alpha = alpha
        self.reduction = reduction
        self.dice = BinaryDiceLoss(reduction=reduction)
        self.wbce = WBCEWithLogitLoss(weight=weight, reduction=reduction)
        self.dice_loss = None
        self.wbce_loss = None

    def forward(self, output, target):
        self.dice_loss = self.dice(output, target)
        self.wbce_loss = self.wbce(output, target)
        loss = self.alpha * self.wbce_loss + self.dice_loss
        return loss


class BinaryFocal_DiceLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=1.0, reduction='mean'):
        super(BinaryFocal_DiceLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        assert 0 <= alpha <= 1, '`alpha` should in [0,1]'
        self.weight = weight
        self.reduction = reduction
        self.dice = BinaryDiceLoss(reduction=reduction)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)

    def forward(self, output, target):
        self.dice_loss = self.dice(output, target)
        self.focal_loss = self.focal(output, target)
        loss = self.weight * self.focal_loss + self.dice_loss
        return loss


class L1DiceLoss(nn.Module):
    def __init__(self, alpha=1.0, reduction='mean'):
        """
        combination of Weight Binary Cross Entropy and Binary Dice Loss
        Args:
            @param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            @param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
            @param alpha: weight between WBCE('Weight Binary Cross Entropy') and binary dice, apply on WBCE
        Shapes:
            output: A tensor of shape [N, *] without sigmoid activation function applied
            target: A tensor of shape same with output
        """
        super(L1DiceLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.alpha = alpha
        self.reduction = reduction
        self.dice = BinaryDiceLoss(reduction=reduction)
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(self, output, target):
        dice_loss = self.dice(output, target)
        l1_loss = self.l1(output, target)
        loss = self.alpha * l1_loss + dice_loss
        return loss
