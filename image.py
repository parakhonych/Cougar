import cv2
from PyQt5.QtWidgets import QMdiSubWindow, QLabel
from PyQt5.QtGui import QImage, QPixmap
class Sub_Image:
    def __init__(self, image_name, image_data, image_number, image_gray):
        self.name = image_name
        self.data = image_data
        self.number = image_number
        self.gray = image_gray
        self.sub = Sub_Window(self.name, self.data, self.number)



class Sub_Window(QMdiSubWindow):
    def __init__(self, img_name, img_data, img_number):
        super().__init__()
        self.title = img_name
        self.sub_data = img_data
        self.number = img_number
        self.image_label = QLabel()
        self.setWidget(self.image_label)
        self.setWindowTitle(self.title)
        rgb_image = cv2.cvtColor(self.sub_data , cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        tempimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap(tempimage)
        self.image_label.setPixmap(self.pixmap.copy())