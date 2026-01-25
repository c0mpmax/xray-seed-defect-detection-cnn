import cv2
import numpy as np
import torch

from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, EnsureType, Activations, AsDiscrete
from torch.utils.data import DataLoader

from utils import SEG_MODEL_PATH
from segmentation.loader import SimplePNGLoader
from segmentation.postprocess import cv_fill, smooth_thicken
from segmentation.model import SeedUNet


class SegmentationService:

    def __init__(self):
        self.model = SeedUNet(SEG_MODEL_PATH)

        self.post_trans = Compose([
            EnsureType(),
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.545),
        ])


    def create_mask(self, image_path):

        original = cv2.imread(image_path)

        if original is None:
            raise ValueError(f"Cannot read image: {image_path}")

        transforms = Compose([
            SimplePNGLoader(),
            EnsureType(),
        ])

        ds = Dataset(data=[{"image": image_path}], transform=transforms)
        loader = DataLoader(ds, batch_size=1)

        # ----UNet----
        for batch in loader:

            img = batch["image"].to(self.model.device)

            outputs = self.model.predict(img)

            outputs = [
                self.post_trans(i)
                for i in decollate_batch(outputs)
            ]

            out = outputs[0]


            mask = out.detach().cpu().numpy()[0]
            mask = (mask > 0).astype(np.uint8)

            # ---- postprocessing with cv2 ----
            mask = cv_fill(mask)
            mask = smooth_thicken(mask)

            mask = cv2.resize(
                mask,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            debug_path = image_path.replace(".png", "_mask.png")
            cv2.imwrite(debug_path, mask * 255)

            return mask


    def debug_save(self, mask, image_path):
        name = image_path.replace(".png", "_mask.png")
        cv2.imwrite(name, mask * 255)

