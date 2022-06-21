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
        self.action_cascade = self.findChild(QAction, "actionCascade")
        self.action_histogram = self.findChild(QAction, "actionHistogram")

        #Click Button
        self.action_open_grayscale.triggered.connect(self.open_windows_gray)
        self.action_open_color.triggered.connect(self.open_windows_color)
        self.action_cascade.triggered.connect(self.mdi.cascadeSubWindows)
        self.action_histogram.triggered.connect(self.show_histogram)

        self.mdi.subWindowActivated.connect(self.update_active_window)

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
        f_paths, _ = QFileDialog.getOpenFileNames(self, "Open file", "D:\\Test_Images\\", "BMP files(*.bmp);;"
                                                                                         "JPEG files (*.jpeg *.jpg);;"
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

    def show_histogram(self):
        if self.active_window.gray == True:
            my_hist = np.zeros(256)
            print(self.active_window.data.shape)

            for h in range(self.active_window.data.shape[0]):
                for w in range(self.active_window.data.shape[1]):
                    current_pixel = self.active_window.data[h, w]
                    # print(current_pixel)
                    my_hist[current_pixel] += 1
            plt.figure(self.active_window.number)
            plt.plot(my_hist)
            plt.show()
        else:
            my_hist = [np.zeros(256), np.zeros(256), np.zeros(256)]

            for w in range(self.active_window.data.shape[0]):
                for h in range(self.active_window.data.shape[1]):
                    for i in range(self.active_window.data.shape[2]):
                        pixel = self.active_window.data[w][h][i]
                        my_hist[i][pixel] += 1
            plt.figure(self.active_window.number)
            plt.plot(my_hist[0], color='b')
            plt.plot(my_hist[1], color='g')
            plt.plot(my_hist[2], color='r')
            plt.show()

    def update_active_window(self, sub):
        if sub.number in self.windows:
            self.active_window = self.windows.get(sub.number)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    UIWindow = UI()
    app.exec_()


