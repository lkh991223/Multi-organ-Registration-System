import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import cv2
import os
import math
import Test
import fillempty
import surface_distance
from PyQt5.QtWidgets import QMessageBox

class ThirdWindow(QMainWindow):
    def __init__(self, bladder_image_path, cervical_image_path, rectum_image_path):
        super().__init__()
        self.bladder_image_path = bladder_image_path
        self.cervical_image_path = cervical_image_path
        self.rectum_image_path = rectum_image_path

        self.bladder_image_path_mov = './results/moving bladder image/158_16_test.png'
        self.cervical_image_path_mov = './results/moving cervical image/158_16_test.png'
        self.rectum_image_path_mov = './results/moving rectum image/158_16_test.png'
        self.image_path_mov = './results/moving image/158_16_test.png'
        self.grid = './results/grid/158_16.png'

        self.setWindowTitle('Registration Results')
        self.setGeometry(100, 100, 700, 500)
        # 配准后各标签
        mov_bladder_label = QLabel('Bladder Fixed Image')
        mov_bladder_label.setAlignment(Qt.AlignCenter)
        mov_bladder_label.setPixmap(QPixmap(self.bladder_image_path_mov).scaled(200, 200, Qt.KeepAspectRatio))
        mov_cervical_label = QLabel('Cervix Fixed Image')
        mov_cervical_label.setAlignment(Qt.AlignCenter)
        mov_cervical_label.setPixmap(QPixmap(self.cervical_image_path_mov).scaled(200, 200, Qt.KeepAspectRatio))
        mov_rectum_label = QLabel('Rectum Fixed Image')
        mov_rectum_label.setAlignment(Qt.AlignCenter)
        mov_rectum_label.setPixmap(QPixmap(self.rectum_image_path_mov).scaled(200, 200, Qt.KeepAspectRatio))

        write_mov_label = QLabel(self)
        write_mov_label.setText('The Registered Label Images of Each Organ are as shown above')
        write_mov_label.setAlignment(Qt.AlignCenter)

        mov_label_layout = QHBoxLayout()
        mov_label_layout.addWidget(mov_bladder_label)
        mov_label_layout.addWidget(mov_cervical_label)
        mov_label_layout.addWidget(mov_rectum_label)

        mov_write_layout = QHBoxLayout()
        mov_write_layout.addWidget(write_mov_label)

        mov_label_write_layout = QVBoxLayout()
        mov_label_write_layout.addLayout(mov_label_layout)
        mov_label_write_layout.addLayout(mov_write_layout)
        
        ref_bladder_label = QLabel('Bladder Fixed Image')
        ref_bladder_label.setAlignment(Qt.AlignCenter)
        ref_bladder_label.setPixmap(QPixmap(self.bladder_image_path).scaled(200, 200, Qt.KeepAspectRatio))
        ref_cervical_label = QLabel('Cervix Fixed Image')
        ref_cervical_label.setAlignment(Qt.AlignCenter)
        ref_cervical_label.setPixmap(QPixmap(self.cervical_image_path).scaled(200, 200, Qt.KeepAspectRatio))
        ref_rectum_label = QLabel('Rectum Fixed Image')
        ref_rectum_label.setAlignment(Qt.AlignCenter)
        ref_rectum_label.setPixmap(QPixmap(self.rectum_image_path).scaled(200, 200, Qt.KeepAspectRatio))

        ref_label_layout = QHBoxLayout()
        ref_label_layout.addWidget(ref_bladder_label)
        ref_label_layout.addWidget(ref_cervical_label)
        ref_label_layout.addWidget(ref_rectum_label)

        write_ref_label = QLabel(self)
        write_ref_label.setText('The Reference Label Images of Each Organ are as shown above')
        write_ref_label.setAlignment(Qt.AlignCenter)

        ref_write_layout = QHBoxLayout()
        ref_write_layout.addWidget(write_ref_label)

        ref_label_write_layout = QVBoxLayout()
        ref_label_write_layout.addLayout(ref_label_layout)
        ref_label_write_layout.addLayout(ref_write_layout)

        re_label = QLabel('Registered Image')
        re_label.setAlignment(Qt.AlignCenter)
        re_label.setPixmap(QPixmap(self.image_path_mov).scaled(300, 300, Qt.KeepAspectRatio))

        write_regist_label = QLabel(self)
        write_regist_label.setText('Registered Image')
        write_regist_label.setAlignment(Qt.AlignCenter)

        grid = QLabel('Grid')
        grid.setAlignment(Qt.AlignCenter)
        grid.setPixmap(QPixmap(self.grid).scaled(300, 300, Qt.KeepAspectRatio))

        write_grid = QLabel(self)
        write_grid.setText('Grid')
        write_grid.setAlignment(Qt.AlignCenter)


        regist_layout = QVBoxLayout()
        regist_layout.addWidget(re_label)
        regist_layout.addWidget(write_regist_label)
        regist_layout.addWidget(grid)
        regist_layout.addWidget(write_grid)

        button = QPushButton('Attain Merics')
        button_layout = QHBoxLayout()
        button_layout.addWidget(button)

        button.clicked.connect(lambda: self.results_calculate())

        ALL_layout = QHBoxLayout()

        main_layout = QVBoxLayout()
        main_layout.addLayout(mov_label_write_layout)
        main_layout.addLayout(ref_label_write_layout)
        main_layout.addLayout(button_layout)

        ALL_layout.addLayout(main_layout)
        ALL_layout.addLayout(regist_layout)




        central_widget = QWidget()
        central_widget.setLayout(ALL_layout)
        self.setCentralWidget(central_widget)

    def hxx(self, x, y):
        # x[x <= 0.00001] = 0
        # y[y <= 0.00001] = 0
        # x[x > 0.00001] = 1
        # y[y > 0.00001] = 1
        x = np.reshape(x, -1)
        y = np.reshape(y, -1)
        # print(x.shape)
        size = x.shape[-1]
        # size = 0
        # for i in range(x.shape[-1]):
        #     if x[i]>=5 and y[i]>=5:
        #         size += 1
        px = np.histogram(x, 256, (0, 255))[0] / size
        py = np.histogram(y, 256, (0, 255))[0] / size
        hx = - np.sum(px * np.log(px + 1e-8))
        hy = - np.sum(py * np.log(py + 1e-8))

        hxy = np.histogram2d(x, y, 256, [[0, 255], [0, 255]])[0]
        hxy /= (1.0 * size)
        hxy = - np.sum(hxy * np.log(hxy + 1e-8))

        r = hx + hy - hxy
        return r

    def ASD(self, moving, fixed, spacing=(1, 1)):
        # moving[moving <= 0.00001] = 0
        # fixed[fixed <= 0.00001] = 0
        # moving[moving > 0.00001] = 1
        # fixed[fixed > 0.00001] = 1
        surface_distances = surface_distance.compute_surface_distances(
            moving > 0.6, fixed > 0.6, spacing_mm=spacing)
        avg_surf_dist = surface_distance.compute_average_surface_distance(surface_distances)
        return 0.5 * (avg_surf_dist[0] + avg_surf_dist[1])

    def dice_score(self, pred, target):
        """This definition generalize to real valued pred and target vector.
    This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """
        pred[pred <= 0.00001] = 0
        target[target <= 0.00001] = 0
        pred[pred > 0.00001] = 1
        target[target > 0.00001] = 1
        top = 2 * np.sum(pred * target)
        union = np.sum(pred + target)
        # eps = np.ones_like(union) * 1e-5
        # bottom = np.max(union, eps)
        dice = np.mean(top / union)
        # print("Dice score", dice)
        return dice

    def results_calculate(self):
        bladder_moving_image = cv2.imread(self.bladder_image_path_mov, 0)
        bladder_fixed_image = cv2.imread(self.bladder_image_path, 0)
        cervical_moving_image = cv2.imread(self.cervical_image_path_mov, 0)
        cervical_fixed_image = cv2.imread(self.cervical_image_path, 0)
        rectum_moving_image = cv2.imread(self.rectum_image_path_mov, 0)
        rectum_fixed_image = cv2.imread(self.rectum_image_path, 0)

        bladder_dice = self.dice_score(bladder_fixed_image, bladder_moving_image)
        bladder_a = self.ASD(bladder_moving_image, bladder_fixed_image)
        bladder_mi = self.hxx(bladder_moving_image, bladder_fixed_image)

        cervical_dice = self.dice_score(cervical_fixed_image, cervical_moving_image)
        cervical_a = self.ASD(cervical_moving_image, cervical_fixed_image)
        cervical_mi = self.hxx(cervical_moving_image, cervical_fixed_image)

        rectum_dice = self.dice_score(rectum_fixed_image, rectum_moving_image)
        rectum_a = self.ASD(rectum_moving_image, rectum_fixed_image)
        rectum_mi = self.hxx(rectum_moving_image, rectum_fixed_image)

        average_dice_bladder = bladder_dice
        average_ASD_bladder = bladder_a
        average_MI_bladder = bladder_mi


        average_dice_cervical = cervical_dice
        average_ASD_cervical = cervical_a
        average_MI_cervical = cervical_mi


        average_dice_rectum = rectum_dice
        average_ASD_rectum = rectum_a
        average_MI_rectum = rectum_mi


        QMessageBox.information(self, 'Metrics Results', "the average_dice_bladder is:" + str(average_dice_bladder) + "\nthe average_MI_bladder is:" + str(average_MI_bladder) + "\nthe average_ASD_bladder is:" + str(average_ASD_bladder) + "the average_dice_cervical is:" + str(average_dice_cervical) + "\nthe average_MI_cervical is:" + str(average_MI_cervical) + "\nthe average_ASD_cervical is:" + str(average_ASD_cervical) + "the average_dice_rectum is:" + str(average_dice_rectum) + "\nthe average_MI_rectum is:" + str(average_MI_rectum) + "\nthe average_ASD_rectum is:" + str(average_ASD_rectum), QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)


class SecondWindow(QMainWindow):
    def __init__(self, ref_fname, mov_fname):
        super().__init__()
        self.ref_fname = ref_fname
        self.mov_fname = mov_fname
        self.setWindowTitle('Processed Images')
        self.setGeometry(100, 100, 500, 300)


        ref_image_label = QLabel('Fixed Image')
        ref_image_label.setAlignment(Qt.AlignCenter)
        ref_image_label.setPixmap(QPixmap(ref_fname).scaled(200, 200, Qt.KeepAspectRatio))

        float_image_label = QLabel('Moving Image')
        float_image_label.setAlignment(Qt.AlignCenter)
        float_image_label.setPixmap(QPixmap(mov_fname).scaled(200, 200, Qt.KeepAspectRatio))

        image_layout = QHBoxLayout()
        image_layout.addWidget(ref_image_label)
        image_layout.addWidget(float_image_label)


        # ref_labels
        self.bladder_image_label = QLabel('Bladder Fixed Label Image')
        self.bladder_image_label.setAlignment(Qt.AlignCenter)
        self.cervical_image_label = QLabel('Cervix Fixed Label Image')
        self.cervical_image_label.setAlignment(Qt.AlignCenter)
        self.rectum_image_label = QLabel('Rectum Fixed Label Image')
        self.rectum_image_label.setAlignment(Qt.AlignCenter)


        select_bladder_button = QPushButton('Select Fixed Bladder Label Image')
        select_cervical_button = QPushButton('Select Fixed CervixLabel Image')
        select_rectum_button = QPushButton('Select Fixed Rectum Label Image')


        select_bladder_button.clicked.connect(lambda: self.chooserefImage('bladder'))
        select_cervical_button.clicked.connect(lambda: self.chooserefImage('cervical'))
        select_rectum_button.clicked.connect(lambda: self.chooserefImage('rectum'))

        
        image_selection_layout = QHBoxLayout()
        image_selection_layout.addWidget(self.bladder_image_label)
        image_selection_layout.addWidget(self.cervical_image_label)
        image_selection_layout.addWidget(self.rectum_image_label)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(select_bladder_button)
        buttons_layout.addWidget(select_cervical_button)
        buttons_layout.addWidget(select_rectum_button)

        self.bladder_image_label_2 = QLabel('Bladder Moving Label Image')
        self.bladder_image_label_2.setAlignment(Qt.AlignCenter)
        self.cervical_image_label_2 = QLabel('Cervix Moving Label Image')
        self.cervical_image_label_2.setAlignment(Qt.AlignCenter)
        self.rectum_image_label_2 = QLabel('Rectum Moving Label Image')
        self.rectum_image_label_2.setAlignment(Qt.AlignCenter)

        select_bladder_button_2 = QPushButton('Select Moving Bladder Label Image')
        select_cervical_button_2 = QPushButton('Select Moving Cervix Label Image')
        select_rectum_button_2 = QPushButton('Select Moving Rectum Label Image')


        select_bladder_button_2.clicked.connect(lambda: self.choosemovImage('bladder'))
        select_cervical_button_2.clicked.connect(lambda: self.choosemovImage('cervical'))
        select_rectum_button_2.clicked.connect(lambda: self.choosemovImage('rectum'))


        image_selection_layout_2 = QHBoxLayout()
        image_selection_layout_2.addWidget(self.bladder_image_label_2)
        image_selection_layout_2.addWidget(self.cervical_image_label_2)
        image_selection_layout_2.addWidget(self.rectum_image_label_2)

        buttons_layout_2 = QHBoxLayout()
        buttons_layout_2.addWidget(select_bladder_button_2)
        buttons_layout_2.addWidget(select_cervical_button_2)
        buttons_layout_2.addWidget(select_rectum_button_2)

        button = QPushButton('Click to start Registration')
        button.clicked.connect(self.register)
        firstbutton = QHBoxLayout()
        firstbutton.addWidget(button)

        
        main_layout = QVBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(image_selection_layout)
        main_layout.addLayout(buttons_layout)
        main_layout.addLayout(image_selection_layout_2)
        main_layout.addLayout(buttons_layout_2)
        main_layout.addLayout(firstbutton)
        # main_layout.addWidget(self.ref_image_label)
        # main_layout.addWidget(self.float_image_label)


        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def chooserefImage(self, image_type):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image files (*.png *.jpeg *.jpg *.bmp *.gif)')
        if fname:
            if image_type == 'bladder':
                self.bladder_image_path = fname
                self.bladder_image_label.setPixmap(QPixmap(fname).scaled(200, 200, Qt.KeepAspectRatio))
            elif image_type == 'cervical':
                self.cervical_image_path = fname
                self.cervical_image_label.setPixmap(QPixmap(fname).scaled(200, 200, Qt.KeepAspectRatio))
            elif image_type == 'rectum':
                self.rectum_image_path = fname
                self.rectum_image_label.setPixmap(QPixmap(fname).scaled(200, 200, Qt.KeepAspectRatio))

    def choosemovImage(self, image_type):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image files (*.png *.jpeg *.jpg *.bmp *.gif)')
        if fname:
            if image_type == 'bladder':
                self.bladder_image_path_2 = fname
                self.bladder_image_label_2.setPixmap(QPixmap(fname).scaled(200, 200, Qt.KeepAspectRatio))
            elif image_type == 'cervical':
                self.cervical_image_path_2 = fname
                self.cervical_image_label_2.setPixmap(QPixmap(fname).scaled(200, 200, Qt.KeepAspectRatio))
            elif image_type == 'rectum':
                self.rectum_image_path_2 = fname
                self.rectum_image_label_2.setPixmap(QPixmap(fname).scaled(200, 200, Qt.KeepAspectRatio))

    def register(self):
        Test.main(self.ref_fname, self.mov_fname, self.bladder_image_path, self.cervical_image_path, self.rectum_image_path, self.bladder_image_path_2, self.cervical_image_path_2, self.rectum_image_path_2)
        fillempty.main()
        self.third_win = ThirdWindow(self.bladder_image_path, self.cervical_image_path, self.rectum_image_path)
        self.close()
        self.third_win.show()

    # def setRefImage(self, image_path):
    #     self.ref_image_label.setPixmap(QPixmap(image_path).scaled(200, 200, Qt.KeepAspectRatio))
    #
    # def setFloatImage(self, image_path):
    #     self.float_image_label.setPixmap(QPixmap(image_path).scaled(200, 200, Qt.KeepAspectRatio))





class FirstWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Image Select and Process')
        self.setGeometry(100, 100, 500, 300)

        self.ref_image_path = ''
        self.float_image_path = ''

        self.ref_image_label = QLabel('Fixed Image')
        self.ref_image_label.setAlignment(Qt.AlignCenter)
        self.float_image_label = QLabel('Moving Image')
        self.float_image_label.setAlignment(Qt.AlignCenter)

        select_ref_button = QPushButton('Select Fixed Image')
        select_float_button = QPushButton('Select Moving Image')
        normalize_button = QPushButton('Normalization and Standardization')


        select_ref_button.clicked.connect(lambda: self.chooseImage('ref'))
        select_float_button.clicked.connect(lambda: self.chooseImage('float'))
        normalize_button.clicked.connect(self.onNormalize)


        image_selection_layout = QVBoxLayout()
        image_selection_layout.addWidget(self.ref_image_label)
        image_selection_layout.addWidget(self.float_image_label)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(select_ref_button)
        buttons_layout.addWidget(select_float_button)
        buttons_layout.addWidget(normalize_button)

        
        main_layout = QVBoxLayout()
        main_layout.addLayout(image_selection_layout)
        main_layout.addLayout(buttons_layout)


        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def chooseImage(self, image_type):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image File (*.png *.jpeg *.jpg *.bmp *.gif)')
        if fname:
            if image_type == 'ref':
                self.ref_image_path = fname
                self.ref_image_label.setPixmap(QPixmap(fname).scaled(200, 200, Qt.KeepAspectRatio))
            elif image_type == 'float':
                self.float_image_path = fname
                self.float_image_label.setPixmap(QPixmap(fname).scaled(200, 200, Qt.KeepAspectRatio))

    def onNormalize(self):
        if self.ref_image_path != '' and self.float_image_path != '':
            self.normalize(self.ref_image_path, self.float_image_path)
       

    def normalize(self, ref_image_path, float_image_path):
        
        self.second_win = SecondWindow(ref_image_path, float_image_path)
        self.close()
        self.second_win.show()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    first_window = FirstWindow()
    first_window.show()
    sys.exit(app.exec_())
