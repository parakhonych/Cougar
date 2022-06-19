import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QMdiArea, QAction

class UI(QMainWindow):
    couter = 0
    def __init__(self):
        super(UI, self).__init__()
        self.windows = dict()
        self.active_window = None

        # Load the ui file
        uic.loadUi("main_window.ui", self)

        # Define our widget mdi
        self.mdi = self.findChild(QMdiArea, "mdiArea")
        self.setCentralWidget(self.mdi)

        #Define File menu
        self.action_open_grayscale = self.findChild(QAction, "actionGrayscale")
        self.action_open_color = self.findChild(QAction, "actionColor")

        self.show()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    UIWindow = UI()
    app.exec_()


