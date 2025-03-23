import argparse

# CHANGE additional imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from oxford_pet import load_dataset
from models.unet import UNet
import torch.optim as optim

def train(args):
    # implement the training function here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> Device(cpu/cuda): {device}")

    # 載入資料集
    train_dataset = load_dataset(args.data_path, mode="train")
    valid_dataset = load_dataset(args.data_path, mode="valid")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # 建立模型
    model = UNet(in_channels=3, out_channels=1)  # 二元分類
    model = model.to(device)

    # 損失函數 & 優化器
    criterion = nn.BCEWithLogitsLoss()  # binary segmentation 通常用這個
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].float().to(device)  # shape: [B, 3, H, W]
            masks = batch["mask"].float().to(device)    # shape: [B, H, W]
            masks = masks.unsqueeze(1)                  # → [B, 1, H, W]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss/len(train_loader):.4f}")

    print("✅ 訓練完成！")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default='./dataset/oxford-iiit-pet', help='path of the input data') # CHANGE add the default path
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs') # CHANGE default epochs 5 to 10
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size') # CHANGE default batch size 1 to 32
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)