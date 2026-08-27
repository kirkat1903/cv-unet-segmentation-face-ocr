"""
Подготовка датасета LGG MRI Segmentation под структуру segmentation_unet.py
-----------------------------------------------------------------------------
Датасет: "Brain MRI segmentation" (LGG Segmentation Dataset)
Оригинал: Mateusz Buda et al., https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
Публичное зеркало без авторизации (используется этим скриптом):
  https://huggingface.co/datasets/gymprathap/Brain-MRI-LGG-Segmentation

Что внутри: МРТ-снимки мозга (FLAIR) 110 пациентов с низкодифференцированной
глиомой + ручные маски опухоли. Задача — бинарная сегментация (опухоль/фон).
Классический учебный датасет для медицинской сегментации.

Оригинальная структура после распаковки:
    kaggle_3m/
      TCGA_CS_4941_19960909/
        TCGA_CS_4941_19960909_1.tif        <- снимок
        TCGA_CS_4941_19960909_1_mask.tif   <- маска (0/255)
        TCGA_CS_4941_19960909_2.tif
        TCGA_CS_4941_19960909_2_mask.tif
        ...
      TCGA_CS_4942_19970222/
        ...
      data.csv

Этот скрипт:
  1. Скачивает .zip (~750 МБ) напрямую с Hugging Face — БЕЗ токена и логина
  2. Распаковывает архив
  3. Делит пациентов на train/val (85/15) — деление ИМЕННО по пациентам,
     чтобы срезы одного мозга не попали одновременно в train и val (утечка данных)
  4. Конвертирует .tif -> .jpg (снимки) и .tif -> .png (маски, бинаризованные)
  5. Раскладывает всё в структуру, которую ожидает segmentation_unet.py:
       data/train/images, data/train/masks, data/val/images, data/val/masks

Зависимости:
  pip install requests pillow numpy tqdm
"""

import zipfile
import random
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


DATASET_URL = (
    "https://huggingface.co/datasets/gymprathap/Brain-MRI-LGG-Segmentation"
    "/resolve/main/Brain-MRI-LGG-Segmentation.zip?download=true"
)
ZIP_PATH = Path("lgg_mri.zip")
RAW_DIR = Path("raw_lgg_mri")           # куда распаковывается сырой датасет
OUT_DIR = Path("data")                  # итоговая структура для обучения
VAL_FRACTION = 0.15
RANDOM_SEED = 42


def download_dataset():
    """Скачивает архив напрямую по прямой ссылке — авторизация не нужна."""
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"Похоже, датасет уже распакован в {RAW_DIR}, пропускаю скачивание.")
        return

    if not ZIP_PATH.exists():
        print("Скачиваю датасет с Hugging Face (~750 МБ, без токена)...")
        response = requests.get(DATASET_URL, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(ZIP_PATH, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Загрузка"
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    else:
        print(f"Архив {ZIP_PATH} уже скачан, пропускаю загрузку.")

    print("Распаковываю архив...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)


def find_patient_dirs() -> list[Path]:
    """Находит папки пациентов (у каждой внутри .tif снимки и маски)."""
    # Реальная структура может быть RAW_DIR/kaggle_3m/<patient>/ или RAW_DIR/<patient>/
    candidates = list(RAW_DIR.glob("**/*_mask.tif"))
    patient_dirs = sorted({p.parent for p in candidates})

    if not patient_dirs:
        raise RuntimeError(
            f"Не нашёл ни одной папки пациента с масками в {RAW_DIR}. "
            "Проверьте, что архив распаковался корректно."
        )
    return patient_dirs


def binarize_mask(mask_path: Path) -> Image.Image:
    """Приводит маску к чистому 0/255 (на случай сжатия/артефактов)."""
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    return Image.fromarray(mask)


def prepare_split(patient_dirs: list[Path], split_name: str):
    images_out = OUT_DIR / split_name / "images"
    masks_out = OUT_DIR / split_name / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    pair_count = 0
    for patient_dir in tqdm(patient_dirs, desc=f"Обработка {split_name}"):
        mask_files = sorted(patient_dir.glob("*_mask.tif"))

        for mask_path in mask_files:
            image_path = Path(str(mask_path).replace("_mask.tif", ".tif"))
            if not image_path.exists():
                continue

            stem = image_path.stem  # уникальное имя среза, например TCGA_CS_4941_19960909_1

            # Снимок: tif -> jpg
            img = Image.open(image_path).convert("RGB")
            img.save(images_out / f"{stem}.jpg", quality=95)

            # Маска: tif -> png, бинаризация
            mask = binarize_mask(mask_path)
            mask.save(masks_out / f"{stem}.png")

            pair_count += 1

    print(f"  {split_name}: {pair_count} пар изображение/маска, "
          f"{len(patient_dirs)} пациентов")


def main():
    download_dataset()

    patient_dirs = find_patient_dirs()
    print(f"Найдено пациентов: {len(patient_dirs)}")

    random.seed(RANDOM_SEED)
    random.shuffle(patient_dirs)

    val_count = max(1, int(len(patient_dirs) * VAL_FRACTION))
    val_patients = patient_dirs[:val_count]
    train_patients = patient_dirs[val_count:]

    prepare_split(train_patients, "train")
    prepare_split(val_patients, "val")

    print("\nГотово! Структура данных подготовлена в ./data/")
    print("Можно запускать: python segmentation_unet.py")


if __name__ == "__main__":
    main()
