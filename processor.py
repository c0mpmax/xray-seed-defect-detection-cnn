import cv2
import os
from PIL import Image, ImageStat

from utils import UNKNOWN_DIR


class SeedProcessor:

    # =====================================================
    # 1. CONTOURS
    # =====================================================

    @staticmethod
    def find_contours_from_mask(mask):

        img = (mask * 255).astype("uint8")

        contours, _ = cv2.findContours(
            img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        return contours


    # =====================================================
    # 2. CONTOUR AREA
    # =====================================================

    @staticmethod
    def get_geometry(cnt):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        return area, perimeter


    # =====================================================
    # 3. CROPPING CONTOUR
    # =====================================================

    @staticmethod
    def crop_and_save(original_img, cnt, index):

        x, y, w, h = cv2.boundingRect(cnt)

        # for classification area
        z = 35

        y1 = max(y - z, 0)
        x1 = max(x - z, 0)

        y2 = min(y + h + z, original_img.shape[0])
        x2 = min(x + w + z, original_img.shape[1])

        cropped = original_img[y1:y2, x1:x2]

        if cropped is None or cropped.size == 0:
            return None, x, y, w, h

        # ---- saving for classificator ----
        filename = os.path.join(UNKNOWN_DIR, f"sample_{index}.png")

        cv2.imwrite(filename, cropped)

        return filename, x, y, w, h


    # =====================================================
    # 4. INFO ABOUT CROPPED IMAGE
    # =====================================================

    @staticmethod
    def get_image_stats(image_path):

        im = Image.open(image_path)

        stats = ImageStat.Stat(im)

        pix_count = stats.count[0]
        lightness = stats.mean[0]

        return pix_count, lightness

