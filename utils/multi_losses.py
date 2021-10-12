import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.binary_losses import BinaryDiceLoss, BinaryTverskyLoss, FocalBinaryTverskyLoss


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, *]
         num_classes: An int of number of class
    Shapes:
        predict: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with predict
    Returns:
        A tensor of shape [N, num_classes, *]
    """

    input_shape = tuple(input.shape)  # (N, H, W, ...)
    new_shape = (input_shape[0], num_classes) + input_shape[1:]
    result = torch.zeros(new_shape).to(input.device)
    result = result.scatter_(1, input.unsqueeze(1).long(), 1)
    return result


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        output: A tensor of shape [N, C, *]
        target: A tensor of same shape with output
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.dice = BinaryDiceLoss(**kwargs)
        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

    def forward(self, output, target):
        target = make_one_hot(target, output.shape[1])
        output = F.softmax(output, dim=1)

        assert output.shape == target.shape, 'output & target shape do not match'
        total_loss = 0

        for i in range(target.shape[1]):
            if i not in self.ignore_index:
                dice_loss = self.dice(output[:, i], target[:, i], use_sigmoid=False)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += (dice_loss)
        loss = total_loss / (target.size(1) - len(self.ignore_index))
        return loss


class WCE_DiceLoss(nn.Module):
    ignore_index: int
    def __init__(self, alpha=1.0, weight=None, ignore_index: int = -100, **kwargs):
        super(WCE_DiceLoss, self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.dice = DiceLoss(weight=weight, ignore_index=ignore_index, **kwargs)
        self.wce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, output, target):
        d_loss = self.dice(output, target)
        c_loss = self.wce(output, target.long())
        return d_loss + self.alpha * c_loss


class Focal_DiceLoss(nn.Module):
    def __init__(self, weight=1.0, alpha=None, gamma=2.0, ignore_index: int = -100, **kwargs):
        super(Focal_DiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.dice = DiceLoss(ignore_index=ignore_index, **kwargs)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)

    def forward(self, output, target):
        d_loss = self.dice(output, target)
        f_loss = self.focal(output, target.long())
        return d_loss + self.weight * f_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.as_tensor([alpha, 1 - alpha])
        if isinstance(alpha,list):
            self.alpha = torch.as_tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class MultiTverskyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(MultiTverskyLoss, self).__init__()
        self.weight = weight
        self.tversky = BinaryTverskyLoss(**kwargs)
        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

    def forward(self, output, target):
        target = make_one_hot(target, output.shape[1])
        output = F.softmax(output, dim=1)

        assert output.shape == target.shape, 'output & target shape do not match'
        total_loss = 0

        for i in range(target.shape[1]):
            if i not in self.ignore_index:
                tversky_loss = self.tversky(output[:, i], target[:, i], use_sigmoid=False)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    tversky_loss *= self.weight[i]
                total_loss += (tversky_loss)
        loss = total_loss / (target.size(1) - len(self.ignore_index))
        return loss


class FocalTverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation adaptive with multi class segmentation
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3, weight=None, ignore_index=None):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight = weight
        self.loss_func = FocalBinaryTverskyLoss(self.alpha, self.beta, self.gamma)
        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

    def forward(self, output, target):
        target = make_one_hot(target, output.shape[1])
        output = F.softmax(output, dim=1)

        assert output.shape == target.shape, 'output & target shape do not match'
        total_loss = 0

        for i in range(target.shape[1]):
            if i not in self.ignore_index:
                class_loss = self.loss_func(output[:, i], target[:, i], use_sigmoid=False)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    class_loss *= self.weight[i]
                total_loss += (class_loss)
        loss = total_loss / (target.size(1) - len(self.ignore_index))
        return loss
