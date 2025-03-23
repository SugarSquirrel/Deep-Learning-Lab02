import argparse

# CHANGE additional imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from oxford_pet import load_dataset
from matplotlib import pyplot as plt
from models.unet import UNet
from utils import dice_score

def train(args):
    # implement the training function here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = load_dataset(args.data_path, mode="train")
    valid_dataset = load_dataset(args.data_path, mode="valid")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

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
            images = batch['image'].float().to(device)
            masks = batch['mask'].float().unsqueeze(1).to(device)
            images = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            args.optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        valid_loss = 0.0
        dice_score = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                images = batch['image'].float().to(device)
                masks = batch['mask'].float().unsqueeze(1).to(device)
                images = images.permute(0, 3, 1, 2)

                outputs = model(images)
                loss = criterion(outputs, masks)
                valid_loss += loss.item() * images.size(0)
                dice_score += dice_score(outputs, masks) * images.size(0)

        avg_valid_loss = valid_loss / len(valid_loader.dataset)
        avg_dice_score = dice_score / len(valid_loader.dataset)
        valid_losses.append(avg_valid_loss)
        dice_scores.append(avg_dice_score)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Valid Loss: {avg_valid_loss:.4f} | Dice Score: {avg_dice_score:.4f}")

    # Plot training results
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

    # Save model
    torch.save(model.state_dict(), "unet_model.pth")
    print("模型已儲存為 unet_model.pth")

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