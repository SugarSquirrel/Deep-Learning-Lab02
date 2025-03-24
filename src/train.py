import argparse

# CHANGE additional imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate
from PIL import Image

def set_seed(seed=42):
    """ 設定所有隨機數生成器的 seed 以確保結果可重現 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 針對多 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 保證 determinism，但可能稍微影響效能

# 設定 Seed，確保 reproducibility
SEED = 42  # 這是你的 Seed，可以根據你的最佳結果修改
set_seed(SEED)

# 資料轉換：影像與遮罩都轉為固定大小的 Tensor
class SegmentationTransform:
    def __init__(self, size=(256, 256)):
        self.image_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __call__(self, image, mask, trimap=None):
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # mask 是 shape [1, H, W]，轉成 numpy 方便處理
        mask_np = mask.numpy()[0]
        binary_mask = (mask_np == 1).astype("float32")  # 只保留 label==1 的部分
        mask = torch.from_numpy(binary_mask).unsqueeze(0)  # 再轉回 tensor
        
        return {
            "image": image,
            "mask": mask
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

    if args.model == 'UNet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    else:
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch = 0
    
    train_losses = []
    valid_losses = []
    dice_scores = []

    # Early Stopping 相關變數
    best_valid_loss = float('inf')  # 初始化為正無窮大
    patience_counter = 0  # 記錄驗證損失未改善的次數

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
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
        avg_valid_loss, avg_dice_score = evaluate(net=model, data=valid_loader, device=device, criterion=criterion)

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
        '''

        valid_losses.append(avg_valid_loss)
        dice_scores.append(avg_dice_score)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Valid Loss: {avg_valid_loss:.4f} | Dice Score: {avg_dice_score:.4f}")
        
        # Early Stopping 檢查
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0  # 重置 patience counter
            torch.save(model.state_dict(), "best_model.pth")
            print(f"> 驗證損失改善，儲存模型為 best_model.pth (Loss: {best_valid_loss:.4f})")
        else:
            patience_counter += 1
            print(f"> 驗證損失未改善 ({patience_counter}/{args.patience})")

        # 停止條件
        if patience_counter >= args.patience:
            print("> Early stopping triggered. 停止訓練。")
            break

    # 儲存模型
    torch.save(model.state_dict(), "unet_model.pth")
    print("模型已儲存為 unet_model.pth")

    # 畫圖保存
    epochs_range = np.arange(1, epoch + 1)
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
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs') # CHANGE default epochs 5 to 15
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size') # CHANGE default batch size 1 to 32
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')  # 新增 Early Stopping 的參數
    parser.add_argument('--model', type=str, default='UNet', help='choose UNet or ResNet34_UNet')  # 新增 Early Stopping 的參數
    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)
