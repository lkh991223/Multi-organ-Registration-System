import voxelmorph2d as vm2d

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from skimage.color import rgb2gray
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
from skimage.transform import resize
from skimage import img_as_ubyte
import multiprocessing as mp
from tqdm import tqdm
import gc
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D


use_gpu = torch.cuda.is_available()

save_image_path = './results/moving image/'
save_Bladder_image_path = './results/moving bladder image/'
save_Cervical_image_path ='./results/moving cervical image/'
save_Rectum_image_path ='./results/moving rectum image/'



#image_path = './new data/threeorgan new data_1,1,5/'
#model_save_path = './new data/use Bladder result/model'

device = torch.device("cuda:{}".format(0))

class VoxelMorph():
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """

    def __init__(self, model, input_dims, is_2d=False, use_gpu=False):
        self.dims = input_dims
        if is_2d:
            self.vm = vm2d
            #self.voxelmorph = vm2d.VoxelMorph2d(input_dims[0] * 2, use_gpu)
        # self.optimizer = optim.SGD(
        #     self.voxelmorph.parameters(), lr=1e-4, momentum=0.99)
        # self.optimizer = optim.SGD(
        #     model.parameters(), lr=1e-4, momentum=0.99)
        # self.params = {'batch_size': 3,
        #                'shuffle': True,
        #                'num_workers': 1,
        #                'worker_init_fn': np.random.seed(42)
         #              }
        self.device = torch.device("cuda:0" if use_gpu else "cpu")

    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError as e:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def forward(self, x):
        self.check_dims(x)
        return voxelmorph(x)

    def calculate_loss(self, y, ytrue, n=9, lamda=0.01, is_training=True):
        loss = self.vm.vox_morph_loss(y, ytrue, n, lamda)
        #loss = self.vm.dice_score(y, ytrue)
        return loss


    def get_test_loss(self, model_ft, id, type, imager, labeler, batch_moving, batch_fixed,
                      Bladder_fixed_image_label, Bladder_moving_image_label, Cervical_fixed_image_label,
                      Cervical_moving_image_label, Rectum_fixed_image_label, Rectum_moving_image_label, n=9,
                      lamda=0.01):
        with torch.set_grad_enabled(False):
            imager = torch.Tensor(imager).cuda().float()
            labeler = torch.Tensor(labeler).cuda().float()
            batch_moving = torch.Tensor(batch_moving).cuda()
            batch_fixed = torch.Tensor(batch_fixed).cuda()
            batch_fixed[batch_fixed <= 0.00001] = 0
            batch_fixed[batch_fixed > 0.00001] = 1
            batch_moving[batch_moving <= 0.00001] = 0
            batch_moving[batch_moving > 0.00001] = 1
            Bladder_fixed_image_label = torch.Tensor(Bladder_fixed_image_label).cuda().float()
            Bladder_moving_image_label = torch.Tensor(Bladder_moving_image_label).cuda().float()
            Cervical_fixed_image_label = torch.Tensor(Cervical_fixed_image_label).cuda().float()
            Cervical_moving_image_label = torch.Tensor(Cervical_moving_image_label).cuda().float()
            Rectum_fixed_image_label = torch.Tensor(Rectum_fixed_image_label).cuda().float()
            Rectum_moving_image_label = torch.Tensor(Rectum_moving_image_label).cuda().float()

            registered_image, deformation_matrix = model_ft(id, imager, labeler,Bladder_fixed_image_label, Cervical_fixed_image_label, Rectum_fixed_image_label, Bladder_moving_image_label, Cervical_moving_image_label, Rectum_moving_image_label)
            bianxing = self.vm.SpatialTransformation(use_gpu=True)

            image_bx = bianxing(batch_moving, deformation_matrix)
            Bladder_moving_image_label_bx = bianxing(Bladder_moving_image_label, deformation_matrix)
            Cervical_moving_image_label_bx = bianxing(Cervical_moving_image_label, deformation_matrix)
            Rectum_moving_image_label_bx = bianxing(Rectum_moving_image_label, deformation_matrix)

            batch_fixed[batch_fixed <= 0.00001] = 0
            batch_fixed[batch_fixed > 0.00001] = 1
            image_bx[image_bx <= 0.00001] = 0
            image_bx[image_bx > 0.00001] = 1

            Bladder_fixed_image_label[Bladder_fixed_image_label <= 0.00001] = 0
            Bladder_fixed_image_label[Bladder_fixed_image_label > 0.00001] = 1
            Bladder_moving_image_label_bx[Bladder_moving_image_label_bx <= 0.00001] = 0
            Bladder_moving_image_label_bx[Bladder_moving_image_label_bx > 0.00001] = 1

            Cervical_fixed_image_label[Cervical_fixed_image_label <= 0.00001] = 0
            Cervical_fixed_image_label[Cervical_fixed_image_label > 0.00001] = 1
            Cervical_moving_image_label_bx[Cervical_moving_image_label_bx <= 0.00001] = 0
            Cervical_moving_image_label_bx[Cervical_moving_image_label_bx > 0.00001] = 1

            Rectum_fixed_image_label[Rectum_fixed_image_label <= 0.00001] = 0
            Rectum_fixed_image_label[Rectum_fixed_image_label > 0.00001] = 1
            Rectum_moving_image_label_bx[Rectum_moving_image_label_bx <= 0.00001] = 0
            Rectum_moving_image_label_bx[Rectum_moving_image_label_bx > 0.00001] = 1

            threeorgan_registered_image = image_bx * registered_image
            threeorgan_fixed_image = batch_fixed * labeler
            Bladder_registered_image = Bladder_moving_image_label_bx * registered_image
            Bladder_fixed_image = Bladder_fixed_image_label * batch_fixed
            Cervical_registered_image = Cervical_moving_image_label_bx * registered_image
            Cervical_fixed_image = Cervical_fixed_image_label * batch_fixed
            Rectum_registered_image = Rectum_moving_image_label_bx * registered_image
            Rectum_fixed_image = Rectum_fixed_image_label * batch_fixed



            io.imsave(save_image_path + id + '_' + type + '.png', img_as_ubyte(registered_image[0].cpu().detach().numpy()))
            io.imsave(save_Bladder_image_path + id + '_' + type + '.png',
                      img_as_ubyte(Bladder_moving_image_label_bx[0].cpu().detach().numpy()))
            io.imsave(save_Cervical_image_path + id + '_' + type + '.png',
                      img_as_ubyte(Cervical_moving_image_label_bx[0].cpu().detach().numpy()))
            io.imsave(save_Rectum_image_path + id + '_' + type + '.png',
                      img_as_ubyte(Rectum_moving_image_label_bx[0].cpu().detach().numpy()))



def main(fixed_path, moving_path, baldder_fixed_path, cervical_fixed_path, rectum_fixed_path, baldder_moving_path, cervical_moving_path, rectum_moving_path):

    '''
    In this I'll take example of FIRE: Fundus Image Registration Dataset
    to demostrate the working of the API.
    '''
    model_ft = vm2d.VoxelMorph2d(1, use_gpu)
    model_ft = nn.DataParallel(model_ft)
    model_ft.cuda()
    model_ft = model_ft.to(device)
    vm = VoxelMorph(model_ft,
        (1, 190, 320), is_2d=True)  # Object of the higher level class
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 1,
              'worker_init_fn': np.random.seed(42)
              }

    # params = {'batch_size': 3,
    #                'shuffle': True,
    #                'num_workers': 1,
    #                'worker_init_fn': np.random.seed(42)
    max_epochs = 100
    lowest_loss = 2000
    now = datetime.now()


    #fn_save = os.path.join('./results/', 'Gen_{}_{}_{}_model.pth'.format('voxelmorph', now_str, 'load'))

    filename = fixed_path.split('.')[0].split("CT")[1]
    imager = np.expand_dims(np.expand_dims(torch.Tensor(rgb2gray(io.imread(moving_path))), axis=-1), axis=0)
    labeler = np.expand_dims(np.expand_dims(torch.Tensor(rgb2gray(io.imread(fixed_path))), axis=-1), axis=0)
    Bladder_fixed_image_label = np.expand_dims(np.expand_dims(torch.Tensor(io.imread(baldder_fixed_path)), axis=-1), axis=0)
    Bladder_moving_image_label = np.expand_dims(np.expand_dims(torch.Tensor(io.imread(baldder_moving_path)), axis=-1), axis=0)
    Cervical_fixed_image_label = np.expand_dims(np.expand_dims(torch.Tensor(io.imread(cervical_fixed_path)),
                                                 axis=-1), axis=0)
    Cervical_moving_image_label = np.expand_dims(np.expand_dims(torch.Tensor(io.imread(cervical_moving_path)),
                                                  axis=-1), axis=0)
    Rectum_fixed_image_label = np.expand_dims(np.expand_dims(torch.Tensor(io.imread(rectum_fixed_path)),
                                                 axis=-1), axis=0)
    Rectum_moving_image_label = np.expand_dims(np.expand_dims(torch.Tensor(io.imread(rectum_moving_path)),
                                                  axis=-1), axis=0)

    model_ft.load_state_dict(torch.load('./model/Gen_voxelmorph_0121-235841_load_model.pth'))
    batch_fixed = Bladder_fixed_image_label + Cervical_fixed_image_label + Rectum_fixed_image_label
    batch_moving = Bladder_moving_image_label + Cervical_moving_image_label + Rectum_moving_image_label

    #model_ft.eval()
        # Transfer to GPU
    vm.get_test_loss(model_ft,  filename, 'test', imager, labeler, batch_moving, batch_fixed, Bladder_fixed_image_label, Bladder_moving_image_label, Cervical_fixed_image_label, Cervical_moving_image_label, Rectum_fixed_image_label, Rectum_moving_image_label)



if __name__ == "__main__":
    main()
