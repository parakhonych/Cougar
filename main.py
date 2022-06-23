import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from image import Sub_Image
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QMainWindow, QApplication, QMdiArea, QAction, QFileDialog, QMessageBox, QDialog, QLabel, QDial, QLineEdit, QPushButton
from ipywidgets import GridspecLayout, BoundedIntText, Layout

class Thresholding(QDialog):
    def __init__(self):
        super(Thresholding, self).__init__()
        uic.loadUi("Thresholding.ui", self)
        self.currently_position = 0
        self.check = False

        ## Define our widgets
        self.dial = self.findChild(QDial, "dial")
        self.label = self.findChild(QLabel, "label")

        self.dial.setRange(0, 255)
        self.dial.valueChanged.connect(self.dialer)
        self.show()

    def dialer(self):
        #Grab the Current dial position
        value = self.dial.value()
        #Set label text
        self.label.setText(f'Current Position: {str(value)}')
        self.currently_position = self.dial.value()
        self.check = True
class InputDialog(QWidget):
    def __init__(self):
        super(InputDialog, self).__init__()
        uic.loadUi("input.ui", self)
        self.valueA = "not yet"
        self.valueB = "not yet"
        self.textBoxA = self.findChild(QLineEdit, "lineEdit")
        self.textBoxB = self.findChild(QLineEdit, "lineEdit_2")

        self.show()

    def ok_click(self):
        pass
        #self.valueA = self.textBoxA.text()
        #self.valueB = self.textBoxB.text()

class UI(QMainWindow):
    counter = 0
    def __init__(self):
        super(UI, self).__init__()
        self.windows = dict()
        self.active_window = None
        self.thresholding_value = -1

        # Load the ui file
        uic.loadUi("main_window.ui", self)

        # Define our widget mdi
        self.mdi = self.findChild(QMdiArea, "mdiArea")
        self.setCentralWidget(self.mdi)

        #Define File menu
        self.action_open_grayscale = self.findChild(QAction, "actionGrayscale")
        self.action_open_color = self.findChild(QAction, "actionColor")
        self.action_save = self.findChild(QAction, "actionSave")

        #Window
        self.action_cascade = self.findChild(QAction, "actionCascade")

        #Operation
        self.action_histogram = self.findChild(QAction, "actionHistogram")
        self.action_stretching = self.findChild(QAction, "actionStretching")
        self.action_equalization = self.findChild(QAction, "actionEqualization")
        self.action_negation = self.findChild(QAction, "actionNegation")
        self.action_set_thresholding = self.findChild(QAction, "actionSet_value")
        self.action_calculate_thresholding = self.findChild(QAction, "action_calculate_Thresholding")
        self.action_posterize = self.findChild(QAction, "actionPosterize")
        self.action_set_kernel = self.findChild(QAction, "actionSet_Kernel")
        self.action_blur_normal = self.findChild(QAction, "actionBlur_Normal")
        self.action_blur_gaussian = self.findChild(QAction, "actionBlur_Gaussian")
        self.action_laplacian = self.findChild(QAction, "actionLaplacian")
        self.action_sobel = self.findChild(QAction, "actionSobel")
        self.action_canny = self.findChild(QAction, "actionCanny")
        self.action_linear_sharpening = self.findChild(QAction, "actionLinear_sharpening")
        self.action_directional_edge_detection = self.findChild(QAction, "actionDirectional_edge_detection")
        self.action_universal_linear_neighborhood_operation = self.findChild(QAction, "actionUniversal_linear_neighborhood_operation")
        self.action_erosion = self.findChild(QAction, "actionErosion")
        self.action_dilation = self.findChild(QAction, "actionDilation")
        self.action_morphological_opening = self.findChild(QAction, "actionMorphological_opening")
        self.action_morphological_closure = self.findChild(QAction, "actionMorphological_closure")
        self.action_skeletonization = self.findChild(QAction, "actionSkeletonization")




        #About
        self.action_autor = self.findChild(QAction, "actionAutor")


        #Click Button File
        self.action_open_grayscale.triggered.connect(self.open_windows_gray)
        self.action_open_color.triggered.connect(self.open_windows_color)
        self.action_save.triggered.connect(self.save_as)

        #Click Button Window
        self.action_cascade.triggered.connect(self.mdi.cascadeSubWindows)

        #Click Button Operation
        self.action_histogram.triggered.connect(self.show_histogram)
        self.action_stretching.triggered.connect(self.stretching)
        self.action_equalization.triggered.connect(self.equalization)
        self.action_negation.triggered.connect(self.negation)
        self.action_set_thresholding.triggered.connect(self.set_value_dial)
        self.action_calculate_thresholding.triggered.connect(self.show_Thresholding)
        self.action_posterize.triggered.connect(self.Posterize)
        self.action_set_kernel.triggered.connect(self.open_input)
        self.action_blur_normal.triggered.connect(self.blur)
        self.action_blur_gaussian.triggered.connect(self.gause_blur)
        self.action_laplacian.triggered.connect(self.laplacian)
        self.action_sobel.triggered.connect(self.sobel)
        self.action_canny.triggered.connect(self.canny)
        self.action_linear_sharpening.triggered.connect(self.liner_shap)
        self.action_directional_edge_detection.triggered.connect(self.directi_edge)
        self.action_universal_linear_neighborhood_operation.triggered.connect(self.universal_line)
        self.action_erosion.triggered.connect(self.erosion)
        self.action_dilation.triggered.connect(self.dilation)
        self.action_morphological_opening.triggered.connect(self.morph_open)
        self.action_morphological_closure.triggered.connect(self.morph_close)
        self.action_skeletonization.triggered.connect(self.skeletonization)



        #Click Button About
        self.action_autor.triggered.connect(self.about)


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
    def save_as(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save file", "entry name",
                                                   "All Files (*);;"
                                                   "Bitmap (*.bmp *.dib);;"
                                                   "Image files (*.jpg *.png *.tif)")

        if not file_path:
            return
        cv2.imwrite(file_path, self.active_window.data)
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
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
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
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
            return
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
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
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

    def set_value_dial(self):
        self.ui = Thresholding()

    def open_input(self):
        self.input_window = InputDialog()

    def blur(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        kernel_size = [7, 7]
        try:
            if self.input_window.textBoxA !="not yet" and self.input_window.textBoxB !="not yet":
                kernel_size = [int(self.input_window.textBoxA.text()), int(self.input_window.textBoxB.text())]

        except:
            pass
        blured_img = cv2.blur(self.active_window.data, kernel_size, borderType=cv2.BORDER_REPLICATE)
        name = "Blured(Normal): " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, blured_img, UI.counter, self.active_window.gray)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()

    def gause_blur(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        kernel_size = (7, 7)
        # stange error !
        '''
        try:
            if self.input_window.textBoxA != "not yet" and self.input_window.textBoxB != "not yet":
                kernel_size = (int(self.input_window.textBoxA.text()), int(self.input_window.textBoxB.text()))
                print(kernel_size)
        except:
            pass
        '''
        sigmaX = 0
        borderType = cv2.BORDER_REPLICATE
        gaussblured_img = cv2.GaussianBlur(self.active_window.data, kernel_size, sigmaX, borderType)
        name = "Blured (Gaussian): " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, gaussblured_img, UI.counter, self.active_window.gray)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()

    def show_Thresholding(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
            return
        if self.ui.currently_position == -1:
            QMessageBox.warning(self, "Wrong value", "Please set the value for the thresholding function first.\n")
            return
        myThresh = self.ui.currently_position
        img_th = np.zeros_like(self.active_window.data)
        # loop through image
        for h in range(self.active_window.data.shape[0]):
            for w in range(self.active_window.data.shape[1]):
                current_pixel = self.active_window.data[h, w]
                # print(current_pixel)
                if (current_pixel > myThresh): img_th[h, w] = 1  # thresholding condition
        img_th=img_th*255
        name = "Thresholding: " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, img_th, UI.counter, True)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()
    def Posterize(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
            return

        myPosterizeBinsNum = 8
        # calc size of binning
        myBins = np.arange(0, 255, np.round(255 / myPosterizeBinsNum))

        # init output image
        img_pstrz = np.zeros_like(self.active_window.data)
        # loop through image
        for h in range(self.active_window.data.shape[0]):
            for w in range(self.active_window.data.shape[1]):
                current_pixel = self.active_window.data[h, w]
                # loop through bins
                for bin in range(myPosterizeBinsNum):
                    # print(myBins[bin])
                    if (current_pixel > myBins[bin]): img_pstrz[h, w] = myBins[bin]  # if inside bin assign value

                if (current_pixel > myBins[-1]): img_pstrz[h, w] = 255  # last bin -> fill with max value
        name = "Posterize: " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, img_pstrz, UI.counter, True)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()
    def laplacian(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        ddepth = cv2.CV_64F  # format obrazu wyjściowego
        ksize = 3  # rozmiar filtra
        img_laplacian = cv2.Laplacian(self.active_window.data, ddepth, ksize, borderType=cv2.BORDER_REPLICATE)

        # cv2_imshow(img_laplacian)
        # plt.figure(figsize=(10,10))
        plt.imshow(img_laplacian, cmap='gray')
        plt.show()
        '''
        name = "Laplacian (Gaussian): " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, img_laplacian, UI.counter, self.active_window.gray)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()
        '''
    def sobel(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        sobelx = cv2.Sobel(self.active_window.data, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(self.active_window.data, cv2.CV_64F, 0, 1,
                           ksize=5)  # parametr trzeci i czwarty określają kierunkowość, najpierw X, potem Y

        frame_sobel = cv2.hconcat((sobelx, sobely))

        # cv2_imshow(frame_sobel)
        # plt.figure(figsize=(10,10))
        plt.imshow(frame_sobel, cmap='gray')
        plt.show()
    def canny(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        # Canny
        # prócz obrazu wejściowego jako argumenty wejściowe podajemy dwa progi, gdzie wartości między tymi progami służą do wyznaczenia pikseli połaczonych następnie w krawędzie
        threshold1 = 100
        threshold2 = 200
        img_canny = cv2.Canny(self.active_window.data, threshold1, threshold2)

        # cv2_imshow(img_canny)
        # plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(img_canny, cv2.COLOR_BGR2RGB))
        plt.show()
    def liner_shap(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
            return
        # wyostrzanie
        #mask_sharp1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        mask_sharp2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #mask_sharp3 = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])

        mask_sharp = mask_sharp2
        print('My mask:')
        print(mask_sharp)
        print('Result of filtering:')
        img_sharp = cv2.filter2D(self.active_window.data, cv2.CV_64F, mask_sharp, borderType=cv2.BORDER_REPLICATE)

        # cv2_imshow(img_sharp)
        # plt.figure(figsize=(10,10))
        plt.imshow(img_sharp, cmap='gray')
        plt.show()
    def directi_edge(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
            return
        # Prewitt dla kiernku NE
        mask_prewittNE = np.array([[0, +1, +1], [-1, 0, +1], [-1, -1, 0]])
        print('My mask:')
        print(mask_prewittNE)
        print('Result of filtering:')
        img_prewitt = cv2.filter2D(self.active_window.data, cv2.CV_64F, mask_prewittNE, borderType=cv2.BORDER_REPLICATE)

        # cv2_imshow(img_prewitt)
        # plt.figure(figsize=(10,10))
        plt.imshow(img_prewitt, cmap='gray')
        plt.show()
    def universal_line(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
            return
        def create_expanded_input(value):
            return BoundedIntText(description='', value=1, min=-100, layout=Layout(height='auto', width='auto'))

        grid = GridspecLayout(3, 3)

        for i in range(3):
            for j in range(3):
                grid[i, j] = create_expanded_input(0);
        grid
        kernel = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                kernel[i, j] = int(grid[i, j].value)

        print('My mask:')
        kernel = np.int64(kernel) / np.sum(kernel)
        print(kernel)
        print('Result of filtering:')
        img_filtered = cv2.filter2D(self.active_window.data, cv2.CV_64F, kernel, borderType=cv2.BORDER_REPLICATE)

        # cv2_imshow(img_filtered)
        # plt.figure(figsize=(10,10))
        plt.imshow(img_filtered, cmap='gray')
        plt.show()
    def erosion(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
            return
        kernel = np.ones((5,5),np.uint8)
        img_erode = cv2.erode(self.active_window.data, kernel, iterations=2, borderType=cv2.BORDER_REPLICATE)
        name = "Erosion: " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, img_erode, UI.counter, self.active_window.gray)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()
    def dilation(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
            return
        kernel = np.ones((5,5),np.uint8)
        img_dilate = cv2.dilate(self.active_window.data, kernel, iterations=2, borderType=cv2.BORDER_REPLICATE)
        name = "Dilation: " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, img_dilate, UI.counter, self.active_window.gray)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()
    def morph_open(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
            return
        kernel = np.ones((5, 5), np.uint8)
        img_open = cv2.morphologyEx(self.active_window.data, cv2.MORPH_OPEN, kernel, borderType = cv2.BORDER_REPLICATE)
        name = "Morphological opening: " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, img_open, UI.counter, self.active_window.gray)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()

    def morph_close(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
            return
        kernel = np.ones((5, 5), np.uint8)
        img_close = cv2.morphologyEx(self.active_window.data, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE)
        name = "Morphological opening: " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, img_close, UI.counter, self.active_window.gray)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()

    def skeletonization(self):
        if self.active_window == None:
            QMessageBox.warning(self, "No active window", "Please select File-> Open first to check this operation.\n")
            return
        if self.active_window.gray != True:
            QMessageBox.warning(self, "Incorrect image type", "This function works only for grayscale images.\n")
            return
        _, img = cv2.threshold(self.active_window.data, 127, 255, 0)
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            im_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            im_temp = cv2.subtract(img, im_open)            #
            im_eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, im_temp)
            img = im_eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        name = "Skeletization: " + self.active_window.name
        UI.counter = UI.counter + 1
        image = Sub_Image(name, skel, UI.counter, self.active_window.gray)
        self.windows[UI.counter] = image
        self.mdi.addSubWindow(image.sub)
        image.sub.show()



    def about(self):
        about_program = """
                                <p style="text-align: center">
                                    <b>Image Processing Algorithms</b><br>
                                    Summary application from lab exercises<br>
                                </p>
                                <table>
                                    <tr><td>Autor:</td>         <td>Volodymyr Parakhonych</td></tr>
                                    <tr><td>Nr Albumu:</td>    <td>18442</td></tr>
                                    <tr><td>WIT grupa:</td>     <td>ID06IO1</td></tr>
                                </table>
                               """

        QMessageBox.information(self, "About", about_program)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    UIWindow = UI()
    app.exec_()


