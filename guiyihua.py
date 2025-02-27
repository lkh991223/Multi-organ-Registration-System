import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np



class ImageWindow(QWidget):
    def __init__(self, ref_fname, mov_fname, image_norm_ref, image_norm_mov):
        super().__init__()
        self.image_norm_ref = ref_fname
        self.image_norm_mov = mov_fname
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 500, 500)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create image display widget
        self.label1 = QLabel('以下为标准化归一化后参考图像和浮动图像:', self)

        self.ref = QLabel()
        self.layout.addWidget(self.ref)
        self.ref.setPixmap(QPixmap(self.image_norm_ref[0]))

        self.mov = QLabel()
        self.layout.addWidget(self.mov)
        self.mov.setPixmap(QPixmap(self.image_norm_mov[0]))

        # image_ref = QImage(self.image_norm_ref.data, self.image_norm_ref.shape[1], self.image_norm_ref.shape[0], self.image_norm_ref.strides[0], QImage.Format_Grayscale8)
        # pixmap = QPixmap.fromImage(image_ref)
        # self.ref = QLabel()
        # self.ref.setPixmap(pixmap)
        # image_mov = QImage(self.image_norm_mov.data, self.image_norm_mov.shape[1], self.image_norm_mov.shape[0], self.image_norm_mov.strides[0],QImage.Format_Grayscale8)
        # pixmap = QPixmap.fromImage(image_mov)
        # self.mov = QLabel()
        # self.mov.setPixmap(pixmap)

        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.ref)
        self.layout.addWidget(self.mov)

        self.hbox_layout1 = QHBoxLayout()
        self.hbox_layout2 = QHBoxLayout()


        # Create buttons and image
        self.image_label1 = QLabel()
        self.layout.addWidget(self.image_label1)
        self.image_label2 = QLabel()
        self.layout.addWidget(self.image_label2)
        self.image_label3 = QLabel()
        self.layout.addWidget(self.image_label3)
        self.layout.setLayout(self.hbox_layout2)
        # self.hbox_layout2.addWidget(self.image_label1)
        # self.hbox_layout2.addWidget(self.image_label2)
        # self.hbox_layout2.addWidget(self.image_label3)

        self.btn1 = QPushButton('上传膀胱标签')
        self.btn1.clicked.connect(self.load_image1)
        # self.layout.addWidget(self.btn1)

        self.btn2 = QPushButton('上传宫颈标签')
        self.btn2.clicked.connect(self.load_image2)
        # self.layout.addWidget(self.btn2)

        self.btn3 = QPushButton('上传直肠标签')
        self.btn3.clicked.connect(self.load_image3)
        # self.layout.addWidget(self.btn3)

        self.hbox_layout1.addWidget(self.btn1)
        self.hbox_layout1.addWidget(self.btn2)
        self.hbox_layout1.addWidget(self.btn3)

        self.central_widget.setLayout(self.hbox_layout1)

        self.label2 = QLabel('请上传参考图像各标签', self)
        self.layout.addWidget(self.label2)

        self.show()

    def load_image1(self):
        self.fname_bladder, _ = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.image_label1.setPixmap(QPixmap(self.fname_bladder))

    def load_image2(self):
        self.fname_cervical, _ = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.image_label2.setPixmap(QPixmap(self.fname_cervical))

    def load_image3(self):
        self.fname_rectum, _ = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.image_label3.setPixmap(QPixmap(self.fname_rectum))

class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        self.ref_label = QLabel(self)  # Reference image label
        self.float_label = QLabel(self)  # Floating image label
        # self.data_ref
        # self.data_mov
        # self.image_norm_ref
        # self.image_norm_mov
        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()
        btn1 = QPushButton('选择参考图像', self)
        btn1.clicked.connect(self.load_ref_image)
        btn2 = QPushButton('选择浮动图像', self)
        btn2.clicked.connect(self.load_float_image)
        btn3 = QPushButton('归一化与标准化', self)
        btn3.clicked.connect(self.normalizestandardize)


        vbox.addWidget(btn1)
        vbox.addWidget(self.ref_label)
        vbox.addWidget(btn2)
        vbox.addWidget(self.float_label)
        vbox.addWidget(btn3)

        self.setLayout(vbox)

        self.setWindowTitle('欢迎使用跨模态多器官配准系统')
        self.setGeometry(300, 300, 600, 400)
        self.show()

    def load_ref_image(self):
        self.ref_fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.ref_label.setPixmap(QPixmap(self.ref_fname[0]))

    def load_float_image(self):
        self.mov_fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.float_label.setPixmap(QPixmap(self.mov_fname[0]))


    def normalizestandardize(self):
        ref_image = cv2.imread(self.ref_fname[0],0)
        mov_image = cv2.imread(self.mov_fname[0],0)
        self.data_ref = np.array(ref_image, dtype=np.float32)
        means = self.data_ref.mean()
        stds = self.data_ref.std()
        # print(type(data),type(means),type(stds))
        self.data_ref -= means
        self.data_ref /= stds
        img_min = np.min(self.data_ref)
        img_max = np.max(self.data_ref)
        self.image_norm_ref = (self.data_ref - img_min) / (img_max - img_min)

        self.data_mov = np.array(mov_image, dtype=np.float32)
        means = self.data_mov.mean()
        stds = self.data_mov.std()
        # print(type(data),type(means),type(stds))
        self.data_mov -= means
        self.data_mov /= stds
        img_min = np.min(self.data_mov)
        img_max = np.max(self.data_mov)
        self.image_norm_mov = (self.data_mov - img_min) / (img_max - img_min)
        self.second_win = ImageWindow(self.ref_fname, self.mov_fname, self.image_norm_ref, self.image_norm_mov)
        self.close()
        self.second_win.show()





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    # win = ImageWindow()
    sys.exit(app.exec_())