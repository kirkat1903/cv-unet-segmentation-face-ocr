"""
Проект: Сегментация изображений с помощью U-Net (PyTorch)
----------------------------------------------------------
Задача: бинарная/многоклассовая сегментация (например, дороги, опухоли на
снимках, дефекты на деталях — под свою задачу нужно только заменить датасет).

Структура:
  1. DoubleConv / U-Net архитектура
  2. Dataset для пар (изображение, маска)
  3. Тренировочный цикл с IoU и Dice метриками
  4. Инференс + визуализация

Зависимости:
  pip install torch torchvision opencv-python numpy matplotlib albumentations
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# 1. Архитектура U-Net
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Два подряд идущих Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
class SegmentationDataset(Dataset):
    """Ожидает структуру:
        images/ЫИМЯ.jpg
        masks/ИМЯ.png   (маска: 0 = фон, 255 = объект)
    """

    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.filenames = sorted(os.listdir(images_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = os.path.join(self.images_dir, name)
        mask_path = os.path.join(self.masks_dir, os.path.splitext(name)[0] + ".png")

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.unsqueeze(0) if mask.dim() == 2 else mask


def get_transforms(train: bool, img_size: int = 256):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# 3. Метрики и обучение
# ---------------------------------------------------------------------------
def dice_coefficient(preds, targets, eps=1e-7):
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * targets).sum()
    return (2 * intersection + eps) / (preds.sum() + targets.sum() + eps)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dice_total = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        dice_total += dice_coefficient(preds, masks).item() * images.size(0)
    return dice_total / len(loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SegmentationDataset("data/train/images", "data/train/masks", get_transforms(True))
    val_ds = SegmentationDataset("data/val/images", "data/val/masks", get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    best_dice = 0.0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_dice = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} | loss={train_loss:.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "unet_best.pth")

    print(f"Готово. Лучший Dice на валидации: {best_dice:.4f}")


if __name__ == "__main__":
    main()
