import torch

def dice_score(pred, target):
    pred = torch.sigmoid(pred)  # 將 logits 轉為概率
    threshold = 0.5
    pred = (pred > threshold).float()  # 二值化
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    return dice.mean().item()
