import torch

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    pred = torch.sigmoid(pred)
    threshold = 0.5
    pred = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    return dice.mean().item()

