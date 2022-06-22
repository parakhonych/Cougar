import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image import Sub_Image
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QMdiArea, QAction, QFileDialog,QMessageBox

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
        self.action_stretching = self.findChild(QAction, "actionStretching")
        self.action_equalization = self.findChild(QAction, "actionEqualization")
        self.action_negation = self.findChild(QAction, "actionNegation")

        #Click Button
        self.action_open_grayscale.triggered.connect(self.open_windows_gray)
        self.action_open_color.triggered.connect(self.open_windows_color)
        self.action_cascade.triggered.connect(self.mdi.cascadeSubWindows)
        self.action_histogram.triggered.connect(self.show_histogram)
        self.action_stretching.triggered.connect(self.stretching)
        self.action_equalization.triggered.connect(self.equalization)
        self.action_negation.triggered.connect(self.negation)

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
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray == True:
            my_hist = np.zeros(256)
            for h in range(self.active_window.data.shape[0]):
                for w in range(self.active_window.data.shape[1]):
                    current_pixel = self.active_window.data[h, w]
                    # print(current_pixel)
                    my_hist[current_pixel] += 1
            plt.figure(self.active_window.name +" "+ str(self.active_window.number)+'h')
            plt.plot(my_hist)
            plt.show()
        else:
            my_hist = [np.zeros(256), np.zeros(256), np.zeros(256)]

            for w in range(self.active_window.data.shape[0]):
                for h in range(self.active_window.data.shape[1]):
                    for i in range(self.active_window.data.shape[2]):
                        pixel = self.active_window.data[w][h][i]
                        my_hist[i][pixel] += 1
            plt.figure(self.active_window.name +" "+ str(self.active_window.number)+'h')
            plt.plot(my_hist[0], color='b')
            plt.plot(my_hist[1], color='g')
            plt.plot(my_hist[2], color='r')
            plt.show()


    def update_active_window(self, sub):
        if sub.number in self.windows:
            self.active_window = self.windows.get(sub.number)

    def stretching(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return

        # define scaling range
        im_min = np.min(self.active_window.data)
        im_max = np.max(self.active_window.data)
        new_max = 255

        # calculate contrast stretching

        img_stretch = np.zeros_like(self.active_window.data)

        for h in range(self.active_window.data.shape[0]):
            for w in range(self.active_window.data.shape[1]):
                current_pixel = self.active_window.data[h, w]
                # print(current_pixel)
                img_stretch[h, w] = ((current_pixel - im_min) * new_max) / (im_max - im_min)
        name = "Stretching: " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, img_stretch, UI.counter, True)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()

    def cumsum(self, a):
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    def equalization(self):
        my_hist = np.zeros(256)
        # loop through image
        for h in range(self.active_window.data.shape[0]):
            for w in range(self.active_window.data.shape[1]):
                # get pixel value
                current_pixel = self.active_window.data[h, w]
                # print(current_pixel)
                my_hist[current_pixel] += 1
                # increase the value of histogram vecor correspondig to current pixel value

        cs = self.cumsum(my_hist)
        cs_m = np.ma.masked_equal(cs, 0)  # For masked array, all operations are performed on non-masked elements.
        cs_min = cs_m.min()
        cs_max = cs_m.max()
        cs = ((cs - cs_min) * 255 )/ (cs_max - cs_min)
        cs = cs.astype('uint8')

        img_eq = cs[self.active_window.data]
        name = "Equaliztion: " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, img_eq, UI.counter, True)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()
        plt.figure(name + " " + str(self.active_window.number) + 'e')
        plt.plot(cs)
        plt.show()

    def negation(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.data.dtype != 'uint8':
            QMessageBox.warning(self, "Wrong Image", "Please select File-> Open first to check this operation.\n")
            return

        img_inv = (255-self.active_window.data)
        name = "Negation: " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, img_inv, UI.counter, True)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()







if __name__ == '__main__':
    app = QApplication(sys.argv)
    UIWindow = UI()
    app.exec_()


