import os
import cv2

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QFont

from utils import RES_HEIGHT, RES_WIDTH
from processor import SeedProcessor
from classifier import SeedClassifier
from segmentation.service import SegmentationService


# ======================================================
#                     DESIGN OF APPLICATION
# ======================================================

class Ui_Dialog(object):

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1920, 1080)
        Dialog.setStyleSheet("background-color: #135487")
        
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(450, 13, 220, 22))
        self.label.setText("Path:")
        self.label.setStyleSheet("color: white")
        self.label.setFont(QFont('Times', 15))

        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(660, 10, 600, 30))

        # ----- ORIGINAL IMAGE -----
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(10, 50, 930, 890))
        self.graphicsView.setStyleSheet("background-color: white")

        # ----- RESULT IMAGE -----
        self.graphicsView2 = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView2.setGeometry(QtCore.QRect(980, 50, 930, 890))
        self.graphicsView2.setStyleSheet("background-color: white")

        # ----- BUTTONS -----
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(245, 950, 465, 50))
        self.pushButton.setStyleSheet("background-color: #68BEF1; border-style: outset;border-width: 5px ; "
                                      "border-color: white; border-radius: 18px; color: white")
        self.pushButton.setFont(QFont('Times', 15,))
        self.pushButton.setText("Load")

        self.pushButton2 = QtWidgets.QPushButton(Dialog)
        self.pushButton2.setGeometry(QtCore.QRect(1225, 950, 465, 50))
        self.pushButton2.setStyleSheet("background-color: #68BEF1; border-style: outset;border-width: 5px ; "
                                      "border-color: white; border-radius: 18px; color: white")
        self.pushButton2.setFont(QFont('Times', 15,))
        self.pushButton2.setText("Search")


# ======================================================
#                    LOGIC PART
# ======================================================

class My_Application(QDialog):

    def __init__(self):
        super().__init__()

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # --- сервисы ---
        self.classifier = SeedClassifier()
        self.segmentator = SegmentationService()

        # --- события ---
        self.ui.pushButton.clicked.connect(self.checkPath)
        self.ui.pushButton2.clicked.connect(self.definition)
        self.ui.pushButton2.setEnabled(False)


    # --------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------
    def checkPath(self):

        image_path = self.ui.lineEdit.text()

        if os.path.isfile(image_path):

            scene = QGraphicsScene(self)

            pixmap = QPixmap(image_path)
            pixmap_res = pixmap.scaled(RES_WIDTH, RES_HEIGHT)

            scene.addItem(QGraphicsPixmapItem(pixmap_res))

            self.ui.graphicsView.setScene(scene)
            self.ui.pushButton2.setEnabled(True)


    # --------------------------------------------------
    # MAIN 
    # --------------------------------------------------
    def definition(self):

        image_path = self.ui.lineEdit.text()

        print("IMAGE PATH:", image_path)

        img = cv2.imread(image_path)

        img_original = img.copy()   # cropped image
        img_draw = img.copy()       # cropped image + classificator_area

        if img is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Cannot open image")
            return


        # =================================================
        # 1. SEGMENTATION (UNET)
        # =================================================

        mask = self.segmentator.create_mask(image_path)

        if mask is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Segmentation failed")
            return


        # =================================================
        # 2. CONTOURS FROM MASK
        # =================================================

        contours = SeedProcessor.find_contours_from_mask(mask)

        j = 1

        for cnt in contours:

            area, perimeter = SeedProcessor.get_geometry(cnt)

            if area < 1000:
                continue


            # =================================================
            # 3. CROP FROM ORIGINAL
            # =================================================

            result = SeedProcessor.crop_and_save(img_original, cnt, j)

            if result[0] is None:
                continue

            filename, x, y, w, h = result


            # =================================================
            # 4. FEATURES
            # =================================================

            pix, lightness = SeedProcessor.get_image_stats(filename)


            # =================================================
            # 5. CLASSIFICATION
            # =================================================

            label = self.classifier.predict_file(filename)

            color = (0, 255, 0) if label == "Normal" else (0, 0, 255)


            # =================================================
            # 6. DRAW RESULT
            # =================================================

            cv2.rectangle(
                img_draw,
                (x-10, y-10),
                (x+w+10, y+h+10),
                color,
                2
            )

            cv2.putText(img_draw, f'S={area:.0f}', (x, y+30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

            cv2.putText(img_draw, f'P={perimeter:.0f}', (x, y+10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

            cv2.putText(img_draw, f'Brightness={lightness:.0f}', (x, y+h+10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

            cv2.putText(img_draw, f'Pix={pix}', (x, y-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

            j += 1


        # =================================================
        # 7. SHOW RESULT
        # =================================================

        temp_result = os.path.join(
            os.path.dirname(image_path),
            "result_view.png"
        )

        cv2.imwrite(temp_result, img_draw)

        scene = QGraphicsScene(self)

        pixmap = QPixmap(temp_result)
        pixmap_res = pixmap.scaled(RES_WIDTH, RES_HEIGHT)

        scene.addItem(QGraphicsPixmapItem(pixmap_res))

        self.ui.graphicsView2.setScene(scene)

        self.ui.pushButton2.setEnabled(False)


