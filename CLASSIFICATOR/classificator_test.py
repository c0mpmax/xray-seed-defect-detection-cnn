import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

loaded = keras.models.load_model("D:/DIPLOM/FINAL_DIPLOM_CODE/CLASSIFICATOR/Model_of_Classificator/")
test_root = ("D:/DIPLOM/Classifier/DATASET/only_bad_samples/")
# test_root = ("D:/DIPLOM/FINAL_DIPLOM_CODE/APPLICATION/temp")
TESTING_DATA_DIR = str(test_root)
IMAGE_SHAPE = (224, 224)

datagen_kwargs = dict(rescale=1. / 255, validation_split=0.99)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    TESTING_DATA_DIR,
    subset="validation",
    shuffle=False,
    target_size=IMAGE_SHAPE)

val_image_batch, val_label_batch = next(iter(valid_generator))
true_label_ids = np.argmax(val_label_batch, axis=-1)

classes = [("Дефектное", 0), ("Нормальное", 1)]
dataset_labels = sorted(classes, key=lambda pair:pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])

tf_model_predictions = loaded.predict(val_image_batch)
predicted_ids = np.argmax(tf_model_predictions, axis=-1)
predicted_labels = dataset_labels[predicted_ids]
print(predicted_labels.dtype)
print(predicted_labels)

plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.3)
for n in range((len(predicted_labels))):
    plt.subplot(8, 4, n+1)
    plt.imshow(val_image_batch[n])
    color = "green" if predicted_ids[n] == true_label_ids[n] else "red"
    plt.title(predicted_labels[n].title(), color=color)
    plt.axis('off')
    _ = plt.suptitle("Результаты классификации (Зелёным: верно, Красным: неверно)")
plt.show()