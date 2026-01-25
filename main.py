import sys
import os
from PyQt5.QtWidgets import QApplication

from gui import My_Application

from utils import MODEL_DIR
print("MODEL DIR:", MODEL_DIR)
print("EXISTS:", os.path.exists(MODEL_DIR))


def main():
    app = QApplication(sys.argv)

    window = My_Application()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()