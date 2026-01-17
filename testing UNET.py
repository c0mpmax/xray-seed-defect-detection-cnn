import torch
import os

# path = os.chdir("C:/Users/User/Desktop/Work/Code/UNET")
# path = os.getcwd()
# print(path)
# model = torch.load(path)
# print(model)
# model.eval()

import logging
import sys
import os
import numpy as np
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import monai
from monai.data import ArrayDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
	EnsureType,
	Compose,
	Activations,
	AsDiscrete,
    SaveImage,
)

epochs=1

def main(dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    dir_img = Path('C:/Users/User/Desktop/Work/Code/UNET/')
    images = sorted(glob(os.path.join(dir_img, "*.png")))

    saver = SaveImage(output_dir="C:/Users/User/Desktop/Work/Code/UNET/", output_ext=".png", output_postfix="seg",
                      dtype = 'np.uint32',separate_folder= False, scale = 255)

    val_ds = ArrayDataset(images)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=2, pin_memory=torch.cuda.is_available())
	post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.95)])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8, 16, 32, 64),
        strides=(2, 2, 2, 2),
        num_res_units=0
    ).to(device)

    model.eval()
    with torch.no_grad():
        val_images = None
        val_labels = None
        val_outputs = None
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            roi_size = (1600, 1600)
            sw_batch_size = 2
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels)

            for val_output, val_image, val_label in zip(val_outputs,val_images,val_labels):
                val_output=val_output.detach().cpu().numpy()
                saver(val_output)

if __name__ == "__main__":
    dir = Path('D:/DIPLOM/Unet/')
    main(dir)


# import torch
# import os
# DATASET_PATH = os.path.join("dataset", "train")
# IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# # define the number of channels in the input, number of classes,
# # and number of levels in the U-Net model
# NUM_CHANNELS = 1
# NUM_CLASSES = 1
# NUM_LEVELS = 3
# NUM_EPOCHS = 1
# BATCH_SIZE = 2
#
# BASE_OUTPUT = "output"
# # define the path to the output serialized model, model training
# # plot, and testing image paths
# MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
# PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
# TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
#
# import cv2
# import numpy as np
#
# def make_predictions(model, imagePath):
# 	# set model to evaluation mode
# 	model.eval()
# 	# turn off gradient tracking
# 	with torch.no_grad():
# 		# load the image from disk, swap its color channels, cast it
# 		# to float data type, and scale its pixel values
# 		image = cv2.imread(imagePath)
# 		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 		image = image.astype("float32") / 255.0
# 		# resize the image and make a copy of it for visualization
# 		image = cv2.resize(image, (128, 128))
# 		# find the filename and generate the path to ground truth
# 		# mask
# 		filename = imagePath.split(os.path.sep)[-1]
# 		groundTruthPath = os.path.join(MASK_DATASET_PATH,
# 			filename)
# 		# load the ground-truth segmentation mask in grayscale mode
# 		# and resize it
# 		gtMask = cv2.imread(groundTruthPath, 0)
# 		gtMask = cv2.resize(gtMask, (INPUT_IMAGE_HEIGHT,
# 			INPUT_IMAGE_HEIGHT))
#
# 		# make the channel axis to be the leading one, add a batch
# 		# dimension, create a PyTorch tensor, and flash it to the
# 		# current device
# 		image = np.transpose(image, (2, 0, 1))
# 		image = np.expand_dims(image, 0)
# 		image = torch.from_numpy(image).to(DEVICE)
# 		# make the prediction, pass the results through the sigmoid
# 		# function, and convert the result to a NumPy array
# 		predMask = model(image).squeeze()
# 		predMask = torch.sigmoid(predMask)
# 		predMask = predMask.cpu().numpy()
#
# imagePaths = open(TEST_PATHS).read().strip().split("\n")
# # load our model from disk and flash it to the current device
# unet = torch.load(MODEL_PATH).to(DEVICE)
# # iterate over the randomly selected test image paths
# for path in imagePaths:
# 	# make predictions and visualize the results
# 	make_predictions(unet, path)

