import sys
import os
import PyQt5
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PIL import Image, ImageStat
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import keras
import numpy as np
import os
import cv2
from glob import glob
from pathlib import Path
import torch
from PIL import ImageOps
from torch.utils.data import DataLoader
import monai
from monai.data import ArrayDataset, decollate_batch, image_reader, PILReader
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Resize,
    LoadImage,
    Compose,
    ScaleIntensity,
    EnsureType,
    SaveImage,
)

path = os.getcwd()
cropped_dir = "temp"
cropped_dir1 = os.path.join(path, cropped_dir)
classificator_dir = "classification"
classificator_dir1 = os.path.join(cropped_dir1, classificator_dir)
if not os.path.exists(cropped_dir1):
    os.makedirs(cropped_dir1)
if not os.path.exists(classificator_dir1):
    os.makedirs(classificator_dir1)

label = None
res_height = 884
res_width = 923

def converter(image):
    image = ImageOps.exif_transpose(image)
    return image.convert("L")

def classificator():
    loaded = keras.models.load_model("C:/Users/User/Desktop/Work/Code/CLASSIFICATOR/Model_of_Classificator/")
    test_root = (cropped_dir1)
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

    classes = [("Defective", 0), ("Normal", 1)]
    dataset_labels = sorted(classes, key=lambda pair: pair[1])
    dataset_labels = np.array([key.title() for key, value in dataset_labels])

    tf_model_predictions = loaded.predict(val_image_batch)
    predicted_ids = np.argmax(tf_model_predictions, axis=-1)
    predicted_labels = dataset_labels[predicted_ids]
    global label
    label = predicted_labels[0]


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1920, 1080)
        Dialog.setStyleSheet("background-color: #135487")

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(450, 13, 220, 22))
        self.label.setObjectName("label")
        self.label.setStyleSheet("color: white")
        self.label.setFont(QFont('bold Times', 15))

        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(660, 10, 600, 30))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setStyleSheet("color: #135487; background-color: white; border-style: outset;border-width: 2px; "
                                    "border-color: white")

        self.lineEdit.setFont(QFont('bold Times', 12))

        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(10, 50, 930, 890))
        # self.graphicsView.setSizeIncrement(QtCore.QSize(0, 0))
        self.graphicsView.setFrameShadow(QtWidgets.QFrame.Raised)
        self.graphicsView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.graphicsView.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView.setStyleSheet("background-color: white")

        self.graphicsView2 = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView2.setGeometry(QtCore.QRect(980, 50, 930, 890))
        # self.graphicsView.setSizeIncrement(QtCore.QSize(0, 0))
        self.graphicsView2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.graphicsView2.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.graphicsView2.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.graphicsView2.setObjectName("graphicsView2")
        self.graphicsView2.setStyleSheet("background-color: white")

        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(245, 950, 465, 50))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setStyleSheet("background-color: #68BEF1; border-style: outset;border-width: 5px ; "
                                      "border-color: white; border-radius: 18px; color: white")
        self.pushButton.setFont(QFont('Times', 15))

        self.pushButton2 = QtWidgets.QPushButton(Dialog)
        self.pushButton2.setGeometry(QtCore.QRect(1225, 950, 465, 50))
        self.pushButton2.setObjectName("pushButton")
        self.pushButton2.setStyleSheet("background-color: #68BEF1; border-style: outset;border-width: 5px ; "
                                      "border-color: white; border-radius: 18px; color: white")
        self.pushButton2.setFont(QFont('Times', 15,))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Search of defective seeds"))
        self.label.setText(_translate("Dialog", "Path:"))
        self.pushButton.setText(_translate("Dialog", "Load"))
        self.pushButton2.setText(_translate("Dialog", "Search"))


class My_Application(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.checkPath)
        self.ui.pushButton2.clicked.connect(self.definition)
        self.ui.pushButton2.setEnabled(False)

    def checkPath(self):
        image_path = self.ui.lineEdit.text()

        if os.path.isfile(image_path):
            scene = QtWidgets.QGraphicsScene(self)
            pixmap = QPixmap(image_path)
            pixmap_res = pixmap.scaled(res_width, res_height)
            item = QtWidgets.QGraphicsPixmapItem(pixmap_res)
            scene.addItem(item)
            self.ui.graphicsView.setScene(scene)
            self.ui.pushButton2.setEnabled(True)

    def definition(self):
        image_path_def = self.ui.lineEdit.text()
        img = cv2.imread(image_path_def)
        if "Photo" in image_path_def:
            red_path_def = image_path_def.replace("Photo", "Mask")
        copy_img = img.copy()
        reds = cv2.imread(red_path_def)
        copy_reds = reds.copy()
        mask = cv2.cvtColor(copy_reds, cv2.COLOR_RGB2GRAY)
        cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        j = 1
        for cnt in cont:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            os.chdir(classificator_dir1)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                z = 35
                if y - z < 0:
                    cropped_img = copy_img[0:y + h + z, x - z:x + w + z]
                elif x - z < 0:
                    cropped_img = copy_img[y:y + h + z, 0:x + w + z]
                else:
                    cropped_img = copy_img[y - z:y + h + z, x - z:x + w + z]
                cv2.imwrite("sample_{}.png".format(j), cropped_img)
                cv2.imwrite("sample_{}.png".format(j+1), cropped_img)
                im = Image.open("sample_{}.png".format(j))
                stats = ImageStat.Stat(im)
                pix = stats.count[0]
                lightness = stats.mean[0]
                classificator()
                if "Normal" in label:
                    cv2.rectangle(img, (x-10, y-10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
                    cv2.putText(img, f'S={area:.{0}f}', ((x - 10) + 4, y + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f'P={perimeter:.{0}f}', ((x - 10) + 4, y + 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f'Яркость={lightness:.{0}f}', ((x - 10) + 4, y + h + 1),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f'Pix={pix}', ((x - 10) + 4, y - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imwrite("saved_info.png", img)
                if "Defective" in label:
                    cv2.rectangle(img, (x-10, y-10), (x + w + 10, y + h + 10), (0, 0, 255), 2)
                    cv2.putText(img, f'S={area:.{0}f}', ((x - 10) + 4, y + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, f'P={perimeter:.{0}f}', ((x - 10) + 4, y + 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, f'Яркость={lightness:.{0}f}', ((x - 10) + 4, y + h + 1),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, f'Pix={pix}', ((x - 10) + 4, y - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imwrite("saved_info.png", img)
            image_path = "saved_info.png"
            scene = QtWidgets.QGraphicsScene(self)
            pixmap = QPixmap(image_path)
            pixmap_res = pixmap.scaled(923, 884)
            item = QtWidgets.QGraphicsPixmapItem(pixmap_res)
            scene.addItem(item)
            self.ui.graphicsView2.setScene(scene)
            self.ui.pushButton2.setEnabled(False)

app = QApplication(sys.argv)
class_instance = My_Application()
class_instance.show()
sys.exit(app.exec_())

