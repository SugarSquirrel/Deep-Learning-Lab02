import argparse

# CHANGE additional imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
# from utils import dice_score
from torchvision import transforms
from oxford_pet import load_dataset
from models.unet import UNet
# from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate

# 資料轉換：影像與遮罩都轉為固定大小的 Tensor
class SegmentationTransform:
    def __init__(self, size=(256, 256)):
        self.image_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __call__(self, image, mask, trimap=None):
        return {
            "image": self.image_transform(image),
            "mask": self.mask_transform(mask)
        }
    
def train(args):
    # implement the training function here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = SegmentationTransform(size=(256, 256))

    # 載入 Dataset 並用 DataLoader 包裝
    train_dataset = load_dataset(data_path=args.data_path, mode="train", transform=transform)
    valid_dataset = load_dataset(data_path=args.data_path, mode="valid", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    valid_losses = []
    dice_scores = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            # 每 N 筆輸出一次訓練進度
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"[Epoch {epoch+1}] Step {i+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 評估模型
        valid_loss, dice_score = evaluate(net=model, data=valid_loader, device=device, criterion=criterion)
        '''
        model.eval()
        valid_loss = 0.0
        dice_score = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                valid_loss += loss.item() * images.size(0)
                dice_score += dice_score(outputs, masks) * images.size(0)

        avg_valid_loss = valid_loss / len(valid_loader.dataset)
        avg_dice_score = dice_score / len(valid_loader.dataset)
        valid_losses.append(avg_valid_loss)
        dice_scores.append(avg_dice_score)
        '''
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Valid Loss: {avg_valid_loss:.4f} | Dice Score: {avg_dice_score:.4f}")

    # 儲存模型
    torch.save(model.state_dict(), "unet_model.pth")
    print("模型已儲存為 unet_model.pth")

    # 畫圖保存
    epochs_range = np.arange(1, args.epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, valid_losses, label="Valid Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, dice_scores, label="Dice Score", color="green")
    plt.title("Dice Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("訓練結果折線圖已儲存為 training_results.png")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default='./dataset/oxford-iiit-pet', help='path of the input data') # CHANGE add the default path
    parser.add_argument('--epochs', '-e', type=int, default=15, help='number of epochs') # CHANGE default epochs 5 to 15
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size') # CHANGE default batch size 1 to 32
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    # parser.add_argument('--optimizer', '-opt', type=str, default='adam', help='optimizer to use')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)