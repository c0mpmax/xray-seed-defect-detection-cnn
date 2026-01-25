import os
from PIL import ImageOps

# =====================================================
# BASE PATH
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================
# DATA FOR CLASSIFICATION
# =====================================================

DATA_DIR = os.path.join(BASE_DIR, "data")

CLASSIFICATION_DIR = os.path.join(DATA_DIR, "classification")

UNKNOWN_DIR = os.path.join(CLASSIFICATION_DIR, "unknown")


# =====================================================
# MODELS
# =====================================================

# (keras SavedModel classificator)
MODEL_DIR = os.path.join(BASE_DIR, "Model_of_Classificator")

# UNet segmentation
SEG_MODEL_DIR = os.path.join(BASE_DIR, "UNET")

SEG_MODEL_PATH = os.path.join(
    SEG_MODEL_DIR,
    "best_metric_model_segmentation2d_array.pth"
)


# =====================================================
# CREATE NECESSARY DIRS
# =====================================================

for d in [DATA_DIR, CLASSIFICATION_DIR, UNKNOWN_DIR]:
    os.makedirs(d, exist_ok=True)


# =====================================================
# CONSTANTS
# =====================================================

RES_HEIGHT = 884
RES_WIDTH = 923


# =====================================================
# UTILS
# =====================================================

def converter(image):

    image = ImageOps.exif_transpose(image)
    return image.convert("L")
