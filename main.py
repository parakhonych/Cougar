import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image import Sub_Image
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QMdiArea, QAction, QFileDialog

class UI(QMainWindow):
    counter = 0
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

        #Click Button
        self.action_open_grayscale.triggered.connect(self.open_windows_gray)
        self.action_open_color.triggered.connect(self.open_windows_color)

        self.show()

    def open_windows_gray(self):
        f_paths, _ = QFileDialog.getOpenFileNames(self, "Open file", "D:\\Test_Images\\", "All Files (*);;"
                                                                                          "BMP files(*.bmp);;"
                                                                                         "JPEG files (*.jpeg *.jpg);;"
                                                                                         "PNG(*.png);;"
                                                                                         "TIFF files (*.tiff *.tif)")
        if not f_paths:
            return
        for f_path in f_paths:

            self.open_window_gray(f_path)

    def open_window_gray(self, fname):
        if fname:
            name = fname.split("/")[-1]
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            UI.counter = UI.counter + 1
            image = Sub_Image(name, img, UI.counter, True)
            self.windows[UI.counter] = image
            self.mdi.addSubWindow(image.sub)
            image.sub.show()
    def open_windows_color(self):
        f_paths, _ = QFileDialog.getOpenFileNames(self, "Open file", "D:\\Test_Images\\", "All Files (*);;"
                                                                                          "BMP files(*.bmp);;"
                                                                                         "JPEG files (*.jpeg *.jpg);;"
                                                                                         "PNG(*.png);;"
                                                                                         "TIFF files (*.tiff *.tif)")
        if not f_paths:
            return
        for f_path in f_paths:

            self.open_window_color(f_path)

    def open_window_color(self, fname):
        if fname:
            name = fname.split("/")[-1]
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            UI.counter = UI.counter + 1
            image = Sub_Image(name, img, UI.counter, False)
            self.windows[UI.counter] = image
            self.mdi.addSubWindow(image.sub)
            image.sub.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    UIWindow = UI()
    app.exec_()


