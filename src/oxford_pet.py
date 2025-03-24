import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=transform):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")
        
        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)
    '''
    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        # if self.transform is not None:
        #     sample = self.transform(**sample)

        # return sample
    '''
    '''
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        # 加載影像和遮罩
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # 應用變換
        if self.transform is not None:
            transformed = self.transform(image, mask)  # 傳入 image 和 mask
            image = transformed["image"]
            mask = transformed["mask"]

        return {"image": image, "mask": mask}
    '''
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        # 讀取影像與原始 trimap（1, 2, 3）
        image = Image.open(image_path).convert("RGB")
        trimap = np.array(Image.open(mask_path))  # numpy array

        # 將 trimap 處理成 binary mask
        mask = self._preprocess_mask(trimap)  # 0 or 1 float32

        # 將 image 和 mask 做相同的轉換
        image = self.transform(image) if self.transform else image
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W] tensor

        return {"image": image, "mask": mask}

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode, transform=None):
    # implement the load dataset function here

    # assert False, "Not implemented yet!"
    # return OxfordPetDataset(root=data_path, mode=mode, transform=transform)
    dataset = OxfordPetDataset(root=data_path, mode=mode, transform=transform)
    return dataset

if __name__ == "__main__":
    
    # download the dataset
    dataset_root = "./dataset/oxford-iiit-pet"

    # check if the dataset is already exist
    # Not yet
    if not os.path.exists(dataset_root):
        print("> 開始下載 Oxford-IIIT Pet Dataset...")
        # download and extract the dataset
        OxfordPetDataset.download(dataset_root)
        print("> 資料集下載完成！")
    
    # Exist
    else:
        # load_dataset Usage
        # view the dataset
        print("> 載入訓練、驗證和測試數據...")
        train_dataset = load_dataset(dataset_root, mode="train")
        valid_dataset = load_dataset(dataset_root, mode="valid")
        test_dataset = load_dataset(dataset_root, mode="test")

        print(f"> 訓練集大小: {len(train_dataset)}")
        print(f"> 驗證集大小: {len(valid_dataset)}")
        print(f"> 測試集大小: {len(test_dataset)}")
        # from torch.utils.data import DataLoader
        # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        print()

        '''show 方法一
        # from PIL import Image
        import numpy as np
        sample = train_dataset[0]
        mask = sample["mask"]
        print(sample.keys())

        
        import torchvision.transforms as T
        img2t = T.ToTensor()
        def trimap2f(trimap):
            return (img2t(trimap) * 255.0 - 1) / 2
        t2img = T.ToPILImage()
        # img = Image.fromarray(mask)
        img = t2img(trimap2f(mask))
        img.show()
        '''
        
        '''show 方法二😍
        from matplotlib import pyplot as plt
        sample = train_dataset[1]
        plt.subplot(1, 2, 1)
        # for visualization we have to transpose back to HWC
        plt.imshow(sample["image"])
        plt.subplot(1, 2, 2)
        # for visualization we have to remove 3rd dimension of mask
        print('> shape mask:', sample["mask"].shape)
        plt.imshow(sample["mask"].squeeze())
        plt.show()
        '''

    '''
        sample = train_dataset[0]
        print(f"> 查看第一個樣本有哪些Key:/n  {sample.keys()}")
        for key, value in sample.items():
            print(f"  {key}: type={type(train_dataset[0][key])} shape={train_dataset[0][key].shape}")

            # show the image
            image = sample[key]
            # transform NumPy to PIL Image
            image = Image.fromarray(image.astype(np.uint8))
            image.show()

        print()
    '''

    '''
    image_path = "D:/user/Desktop/FCU大學生活/大四的/下學期/TAICA-DeepLearning/Lab/Lab02/dataset/oxford-iiit-pet/images/american_bulldog_199.jpg"
    # show
    image = Image.open(image_path)
    # transform NumPy to PIL Image
    # image.show()

    # 檢查色彩通道
    image_array = np.array(image)
    print(f"Image shape: {image_array.shape}")  # (高度, 寬度, 通道數)
    if len(image_array.shape) == 3:
        print("Color channels: ", image_array.shape[2])
    else:
        print("This image is grayscale (single channel).")
    '''

    '''
    from torch.utils.data import DataLoader
    from oxford_pet import OxfordPetDataset

    # 設定資料集路徑
    dataset_root = "./dataset/oxford-iiit-pet"

    # 初始化訓練資料集
    train_dataset = OxfordPetDataset(root=dataset_root, mode="train")

    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 迭代資料
    # for batch in train_loader:
    #     images, masks = batch["image"], batch["mask"]
        # 在此處進行模型訓練或其他操作
    '''