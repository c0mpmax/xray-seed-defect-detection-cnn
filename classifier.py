import tensorflow as tf
import numpy as np
from tensorflow import keras


from utils import MODEL_DIR, CLASSIFICATION_DIR


class SeedClassifier:

    def __init__(self):
        # load model for classification
        self.model = keras.models.load_model(MODEL_DIR)

        # classes for detection
        self.classes = np.array(["Defective", "Normal"])

    def predict_file(self, image_path):

        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(224, 224)
        )

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)

        predicted_id = np.argmax(predictions, axis=-1)[0]

        return self.classes[predicted_id]
