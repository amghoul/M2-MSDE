import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
from . import preprocess
from . import listflowfile as lt
from . import readpfm as rp
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB'), path


def disparity_loader(path):
    return rp.readPFM(path)

def occ_loader(path):
    return Image.open(path)

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity,left_disp_occ, training, normalize,loader=default_loader, dploader=disparity_loader, occloader = occ_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.disp_L_occ = left_disp_occ
        self.loader = loader
        self.dploader = dploader
        self.occloader = occloader
        self.training = training
        self.normalize = normalize

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        disp_L_occ = self.disp_L_occ[index]

        left_img, left_path = self.loader(left)
        right_img, right_path = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        dataL_occ = self.occloader(disp_L_occ)
        dataL_occ = np.ascontiguousarray(dataL_occ, dtype=np.float32)        


        processed = preprocess.get_transform(augment=False, normalize=self.normalize)
        left_img = processed(left_img)
        right_img = processed(right_img)
        temp_tensor = torch.empty(1, 1)
        
        return left_img, right_img, dataL,dataL_occ,temp_tensor,left_path,right_path

    def __len__(self):
        return len(self.left)
