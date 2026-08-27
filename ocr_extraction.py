"""
Проект: OCR — извлечение текста из изображений/сканов документов
--------------------------------------------------------------------
Пайплайн:
  1. Предобработка изображения (OpenCV): выравнивание, бинаризация, шумоподавление
  2. Распознавание текста через EasyOCR (поддерживает русский из коробки)
  3. Постобработка: сортировка блоков текста по положению на странице
  4. Экспорт результата в .txt и .json (с координатами боксов)

Зависимости:
  pip install easyocr opencv-python numpy
"""

import json
import cv2
import numpy as np
import easyocr


class DocumentOCR:
    def __init__(self, languages=("ru", "en"), gpu: bool = True):
        self.reader = easyocr.Reader(list(languages), gpu=gpu)

    # -- предобработка -----------------------------------------------------
    @staticmethod
    def preprocess(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Убираем шум, сохраняя границы текста
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Адаптивная бинаризация — устойчива к неравномерному освещению скана
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, C=15,
        )
        return binary

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Выравнивает наклон отсканированного текста."""
        coords = np.column_stack(np.where(image < 128))
        if coords.size == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    # -- распознавание -----------------------------------------------------
    def extract(self, image_path: str, preprocess: bool = True):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Не удалось открыть файл: {image_path}")

        if preprocess:
            processed = self.preprocess(image)
            processed = self.deskew(processed)
            image_for_ocr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            image_for_ocr = image

        results = self.reader.readtext(image_for_ocr)

        blocks = []
        for bbox, text, confidence in results:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            blocks.append({
                "text": text,
                "confidence": round(float(confidence), 3),
                "bbox": {
                    "x_min": min(x_coords), "x_max": max(x_coords),
                    "y_min": min(y_coords), "y_max": max(y_coords),
                },
            })

        # Сортировка сверху вниз, слева направо — как обычно читают документ
        blocks.sort(key=lambda b: (round(b["bbox"]["y_min"] / 20), b["bbox"]["x_min"]))
        return blocks

    # -- экспорт -----------------------------------------------------
    @staticmethod
    def save_txt(blocks: list, out_path: str):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(b["text"] for b in blocks))

    @staticmethod
    def save_json(blocks: list, out_path: str):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)


def main():
    ocr = DocumentOCR(languages=("ru", "en"), gpu=False)

    image_path = "data/sample_document.jpg"
    blocks = ocr.extract(image_path, preprocess=True)

    print(f"Найдено блоков текста: {len(blocks)}")
    for b in blocks:
        print(f"[{b['confidence']:.2f}] {b['text']}")

    ocr.save_txt(blocks, "output.txt")
    ocr.save_json(blocks, "output.json")
    print("Результат сохранён в output.txt и output.json")


if __name__ == "__main__":
    main()
