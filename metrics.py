import torch


def _zero_when_empty(numerator, denominator):
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))

def dice_coeff(pred, target):
    smooth = 0.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum(-1).float()

    denominator = m1.sum(-1) + m2.sum(-1) + smooth
    return _zero_when_empty(2. * intersection + smooth, denominator)


def iou(pred, target):
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum(-1).float()
    union = m1.sum(-1) + m2.sum(-1) - intersection
    return _zero_when_empty(intersection, union)

def accuracy(pred, target):
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)
    correct = (m1 == m2).sum(-1).float()
    total = m1.size(-1)
    if total == 0:
        return torch.zeros_like(correct)
    return correct / total



def macro_dice_coeff(pred, target):
    num_class = pred.size(1)
    dice = 0
    for i in range(num_class):
        dice += dice_coeff(pred[:, i, :, :], target[:, i, :, :])
    return dice / num_class

def macro_iou(pred, target):
    num_class = pred.size(1)
    iou_score = 0
    for i in range(num_class):
        iou_score += iou(pred[:, i, :, :], target[:, i, :, :])
    return iou_score / num_class
