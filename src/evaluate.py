import torch
from utils import dice_score

def evaluate(net, data, device, criterion):
    # implement the evaluation function here
    # valid_losses = []
    # dice_scores = []

    net.eval()
    valid_loss = 0.0
    total_dice_score  = 0.0
    with torch.no_grad():
        for batch in data:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = net(images)
            loss = criterion(outputs, masks)
            valid_loss += loss.item() * images.size(0)
            total_dice_score  += dice_score(outputs, masks) * images.size(0)

    avg_valid_loss = valid_loss / len(data.dataset)
    avg_dice_score = total_dice_score  / len(data.dataset)

    return avg_valid_loss, avg_dice_score
