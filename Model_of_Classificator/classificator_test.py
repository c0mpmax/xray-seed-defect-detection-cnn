import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

MODEL_PATH = "C:/Users/User/Desktop/Work/Code/Model_of_Classificator/"
TEST_DIR = "C:/Users/User/Desktop/Work/Code/bad_samples/"
IMAGE_SHAPE = (224, 224)

CLASSES = ["Нормальное", "Дефектное"]  # проверь порядок классов

print("Загрузка модели...")
model = keras.models.load_model(MODEL_PATH)

images = []
filenames = []

for file in os.listdir(TEST_DIR):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(TEST_DIR, file)

        img = Image.open(path).convert("RGB")   # важно!
        img = img.resize(IMAGE_SHAPE)
        img = np.array(img) / 255.0

        images.append(img)
        filenames.append(file)

images = np.array(images)

print("Загружено изображений:", len(images))

predictions = model.predict(images)
pred_ids = np.argmax(predictions, axis=-1)
confidences = np.max(predictions, axis=-1)

pred_labels = [CLASSES[i] for i in pred_ids]

print("\n===== РЕЗУЛЬТАТЫ =====")
for name, label, conf in zip(filenames, pred_labels, confidences):
    print(f"{name:20s} -> {label:10s} ({conf:.3f})")

# визуализация
plt.figure(figsize=(12, 10))
cols = 4
rows = int(np.ceil(len(images) / cols))

for i in range(len(images)):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i])
    color = "green" if pred_ids[i] == 0 else "red"
    plt.title(f"{pred_labels[i]} ({confidences[i]:.2f})", color=color)
    plt.axis("off")

plt.suptitle("Проверка модели на дефектных изображениях")
plt.tight_layout()
plt.show()
