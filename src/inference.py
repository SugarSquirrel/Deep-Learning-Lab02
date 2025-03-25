import argparse
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sys
import os
from torchvision.transforms import ToPILImage
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(project_root)
from models.unet import UNet
from oxford_pet import load_dataset
from utils import dice_score
from train import SegmentationTransform
from models.resnet34_unet import ResNet34_UNet
# from torchvision import transforms

# random seed
random.seed()

# 計算整組測試資料集的平均 Dice Score
def calculate_average_dice_score(model, dataset, device):
    model.eval()  # 設定模型為評估模式
    total_dice_score = 0.0
    num_samples = len(dataset)

    with torch.no_grad():
        for sample in dataset:
            # 取得影像和 Ground Truth Mask
            image = sample["image"].unsqueeze(0).to(device)  # 加上 batch 維度
            mask = sample["mask"].unsqueeze(0).to(device)  # 加上 batch 維度

            # 推論
            pred = model(image)
            pred = torch.sigmoid(pred)
            pred_mask = (pred > 0.5).float()

            # 計算 Dice Score
            dice = dice_score(pred_mask, mask)
            total_dice_score += dice

    # 計算平均 Dice Score
    average_dice_score = total_dice_score / num_samples
    return average_dice_score

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', type=str, default='unet', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default='./dataset/oxford-iiit-pet', help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # assert False, "Not implemented yet!"
    transform = SegmentationTransform(size=(256, 256))

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    if args.model.lower() == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
        model.load_state_dict(torch.load("unet_model.pth", map_location=device, weights_only=True))
    else:
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
        model.load_state_dict(torch.load("resnet34_unet_model.pth", map_location=device, weights_only=True))
    model.eval()

    # load 5 random test data
    dataset = load_dataset(data_path=args.data_path, mode="test", transform=transform)
    dataset_length = len(dataset)
    num_samples = 5

    if dataset_length < num_samples:
        # 如果資料不足 5 筆，從頭開始補足
        indices = list(range(dataset_length)) + list(range(num_samples - dataset_length))
    else:
        # 隨機選取 5 筆資料
        indices = random.sample(range(dataset_length), num_samples)

    # 取得隨機選取的資料
    samples = [dataset[i] for i in indices]
    # image = sample["image"]  # 確保只拿到 image

    # type(image) is tensor of shape (3, 256, 256)
    # input_image = image.unsqueeze(0).to(device)  # 加上 batch 維度

    # inference
    # inference for 5 random samples
    # 建立一個 3x5 的子圖 (3 行, 5 列)
    fig, axes = plt.subplots(5, 4, figsize=(7, 8))  # 調整 figsize 以適應 5 筆資料

    for i, sample in enumerate(samples):  # 假設 samples 是隨機選取的 5 筆資料
        image = sample["image"]  # 確保只拿到 image
        input_image = image.unsqueeze(0).to(device)  # 加上 batch 維度

        # 推論
        with torch.no_grad():
            pred = model(input_image)
            pred = torch.sigmoid(pred)
            pred_mask = (pred > 0.5).float()

        # Tensor → Numpy
        input_image_np = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
        pred_mask_np = pred_mask.squeeze().cpu().numpy()

        # 如果有 Ground Truth Mask
        mask = sample["mask"].to(device)
        mask_np = np.array(ToPILImage()(mask).resize((256, 256), resample=Image.NEAREST))
        
        # 計算 Dice Score
        total_dice_score = dice_score(pred_mask, mask.unsqueeze(0))
        
        # intersection = (pred_mask.squeeze() * torch.tensor(mask_np, device=device)).sum()
        # dice_score = (2.0 * intersection) / (pred_mask.sum() + torch.tensor(mask_np, device=device).sum() + 1e-8)
        # dice_score = dice_score.item()  # 轉為 Python float

        # 顯示原始影像
        axes[i, 0].imshow(input_image_np)
        axes[i, 0].set_title(f"Input Image {i+1}", fontsize=8)
        axes[i, 0].axis("off")

        # 顯示 Ground Truth Mask
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f"Ground Truth {i+1}", fontsize=8)
        axes[i, 1].axis("off")

        # 顯示預測結果
        axes[i, 2].imshow(pred_mask_np, cmap='gray')
        axes[i, 2].set_title(f"Predicted Mask {i+1}", fontsize=8)
        axes[i, 2].axis("off")

        # 顯示 Dice Score
        axes[i, 3].text(0.5, 0.5, f"Dice score:\n{total_dice_score:.4f}", fontsize=10, ha='center', va='center')
        axes[i, 3].axis("off")

    # 調整子圖間距
    plt.subplots_adjust(wspace=0.4, hspace=0.6)  # 增加子圖之間的水平和垂直間距
    plt.tight_layout()
    plt.show()

    # 計算整組測試資料集的平均 Dice Score
    average_dice_score = calculate_average_dice_score(model, dataset, device)
    print(f"Average Dice Score for the test dataset: {average_dice_score:.4f}")


    # # ✅ 計算 Dice Score
    # criterion = nn.BCEWithLogitsLoss()
    # avg_valid_loss, avg_dice_score = evaluate(net=model, data=dataset, device=device, criterion=criterion)
    # print(f"Dice Score: {avg_dice_score:.4f}")
