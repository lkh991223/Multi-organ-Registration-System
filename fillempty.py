import cv2
import os
import numpy as np
import skimage
from skimage.measure import label, regionprops

bladder_image_path_mov = './results/moving bladder image/158_16_test.png'
cervical_image_path_mov = './results/moving cervical image/158_16_test.png'
rectum_image_path_mov = './results/moving rectum image/158_16_test.png'


def remove_small_points(binary_img, threshold_area):
    """
    消除二值图像中面积小于某个阈值的连通域(消除孤立点)
    args:
        binary_img: 二值图
        threshold_area: 面积条件大小的阈值,大于阈值保留,小于阈值过滤
    return:
        resMatrix: 消除孤立点后的二值图
    """
    # 输出二值图像中所有的连通域
    img_label, num = label(binary_img, connectivity=1, background=0,
                           return_num=True)  # connectivity=1--4  connectivity=2--8
    # print('+++', num, img_label)
    # 输出连通域的属性，包括面积等
    props = regionprops(img_label)
    ## adaptive threshold
    # props_area_list = sorted([props[i].area for i in range(len(props))])
    # threshold_area = props_area_list[-2]
    resMatrix = np.zeros(img_label.shape).astype(np.uint8)
    for i in range(0, len(props)):
        print('--', props[i].area)
        if props[i].area > threshold_area:
            tmp = (img_label == i + 1).astype(np.uint8)
            # 组合所有符合条件的连通域
            resMatrix += tmp
    resMatrix *= 255

    return resMatrix

def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out

def main():
    img1 = cv2.imread(bladder_image_path_mov, 0)

    img1[img1 <= 0] = 0
    img1[img1 > 0] = 255

    img1 = FillHole(img1)
    img1 = remove_small_points(img1, 80)

    cv2.imwrite(bladder_image_path_mov, img1)

    img2 = cv2.imread(cervical_image_path_mov, 0)

    img2[img2 <= 0] = 0
    img2[img2 > 0] = 255

    img2 = FillHole(img2)
    img2 = remove_small_points(img2, 80)

    cv2.imwrite(cervical_image_path_mov, img2)

    img3 = cv2.imread(rectum_image_path_mov, 0)

    img3[img3 <= 0] = 0
    img3[img3 > 0] = 255

    img3 = FillHole(img3)
    img3 = remove_small_points(img3, 80)

    cv2.imwrite(rectum_image_path_mov, img3)
