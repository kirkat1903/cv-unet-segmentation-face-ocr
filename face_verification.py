"""
Проект: Верификация лиц (Face Verification) на facenet-pytorch + OpenCV
------------------------------------------------------------------------
Задача: по фото человека определить, "тот же это человек или нет"
(1:1 верификация) — например, для системы контроля доступа.

Пайплайн:
  1. MTCNN — детекция и выравнивание лица
  2. InceptionResnetV1 (предобучен на VGGFace2) — получение эмбеддинга 512-d
  3. Сравнение эмбеддингов по косинусному расстоянию
  4. "База" зарегистрированных лиц хранится как словарь {имя: эмбеддинг}

Зависимости:
  pip install facenet-pytorch opencv-python torch numpy
"""

import os
import pickle
import numpy as np
import torch
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DB_PATH = "face_db.pkl"
THRESHOLD = 0.6  # порог косинусного расстояния, подбирается на своих данных


class FaceVerifier:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE, keep_all=False)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
        self.db = self._load_db()

    # -- работа с базой -----------------------------------------------------
    def _load_db(self) -> dict:
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_db(self):
        with open(DB_PATH, "wb") as f:
            pickle.dump(self.db, f)

    # -- эмбеддинги -----------------------------------------------------
    def get_embedding(self, image_bgr: np.ndarray):
        """Возвращает 512-мерный эмбеддинг лица или None, если лицо не найдено."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        face_tensor = self.mtcnn(rgb)

        if face_tensor is None:
            return None

        with torch.no_grad():
            embedding = self.resnet(face_tensor.unsqueeze(0).to(DEVICE))

        return embedding.cpu().numpy().flatten()

    # -- регистрация нового человека -----------------------------------
    def register(self, name: str, image_bgr: np.ndarray) -> bool:
        embedding = self.get_embedding(image_bgr)
        if embedding is None:
            print(f"[{name}] лицо не обнаружено, регистрация не выполнена")
            return False

        self.db[name] = embedding
        self._save_db()
        print(f"[{name}] успешно зарегистрирован")
        return True

    # -- верификация 1:1 -------------------------------------------------
    def verify(self, image_bgr: np.ndarray, claimed_name: str):
        if claimed_name not in self.db:
            return False, None

        embedding = self.get_embedding(image_bgr)
        if embedding is None:
            return False, None

        distance = cosine_distance(embedding, self.db[claimed_name])
        is_match = distance < THRESHOLD
        return is_match, distance

    # -- идентификация 1:N (кто это?) ------------------------------------
    def identify(self, image_bgr: np.ndarray):
        embedding = self.get_embedding(image_bgr)
        if embedding is None or not self.db:
            return None, None

        best_name, best_dist = None, float("inf")
        for name, db_embedding in self.db.items():
            dist = cosine_distance(embedding, db_embedding)
            if dist < best_dist:
                best_name, best_dist = name, dist

        if best_dist < THRESHOLD:
            return best_name, best_dist
        return None, best_dist


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b))


def run_webcam_demo():
    """Демо: живой поток с веб-камеры, идентификация лица в реальном времени."""
    verifier = FaceVerifier()
    cap = cv2.VideoCapture(0)

    print("Нажмите 'r' чтобы зарегистрировать лицо, 'q' — выход")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        name, dist = verifier.identify(frame)
        label = f"{name} ({dist:.2f})" if name else "Неизвестно"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Verification Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            new_name = input("Введите имя для регистрации: ")
            verifier.register(new_name, frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()
