import torch
from torch.autograd import Function
import torch.nn as nn
import numpy as np


# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

class DiceLoss(nn.Module):
    def __init__(self, num_class=5):
        super(DiceLoss, self).__init__()
        self.num_class = num_class

    def forward(self, pred, target, save=True):
        ''' target, pred: (N, H, W, D) '''

        gt_image, pre_image = target.data.cpu(), pred.data.cpu().argmax(1)
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].type(torch.LongTensor) + pre_image[mask]
        count = torch.bincount(label, minlength=self.num_class ** 2)
        self.confusion_matrix = torch.reshape(count, (self.num_class, self.num_class))

        return torch.diag(self.confusion_matrix) * 2 / \
               (torch.sum(self.confusion_matrix, 1) + torch.sum(self.confusion_matrix, 0))

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect / (union * union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
        return grad_input, None
