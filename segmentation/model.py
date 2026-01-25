import torch
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference


class SeedUNet:
    def __init__(self, model_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
            num_res_units=0,
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.eval()

    def predict(self, image):
        with torch.no_grad():
            return sliding_window_inference(
                image,
                roi_size=(1280, 1280),
                sw_batch_size=2,
                predictor=self.model
            )
