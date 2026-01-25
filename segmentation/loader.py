import numpy as np
from PIL import Image

class SimplePNGLoader:
    def __call__(self, data):
        path = data["image"]

        img = Image.open(path).convert("L")
        img = np.array(img).astype(np.float32)

        img = img / 255.0
        img = np.expand_dims(img, 0)

        data["image"] = img
        data["path"] = path
        return data
