import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess 

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB'), path

def disparity_loader(path):
    return Image.open(path)

############kitti 2012
class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity,left_disp_noc, training, flip_vertical, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.disp_L_noc = left_disp_noc
        self.flip_vertical=flip_vertical
        #self.flip_ver = transforms.RandomVerticalFlip()

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]
        disp_L_noc=self.disp_L_noc[index]

        left_img, left_path = self.loader(left)
        right_img, right_path = self.loader(right)
        dataL = self.dploader(disp_L)
        dataL_noc = self.dploader(disp_L_noc)

        if self.training:  
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            dataL_noc = np.ascontiguousarray(dataL_noc,dtype=np.float32)/256
            dataL_noc = dataL_noc[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)  
            left_img   = processed(left_img)
            right_img  = processed(right_img)

            #FLIP_VERTICALLY:
            probability = np.random.random()
            if self.flip_vertical == 1 and probability > 0.5:
                left_img = left_img.flip(-2)
                right_img = right_img.flip(-2)
                dataL = np.flip(dataL, -2)
                dataL_noc = np.flip(dataL_noc, -2)

            temp_tensor = torch.empty(1, 1)
            return left_img, right_img, dataL,dataL_noc,temp_tensor,left_path,right_path
        else:
            w, h = left_img.size

            left_img = left_img.crop((w-1232, h-368, w, h))
            right_img = right_img.crop((w-1232, h-368, w, h))
            w1, h1 = left_img.size

            dataL = dataL.crop((w-1232, h-368, w, h))
            dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
            dataL_noc = dataL_noc.crop((w-1232, h-368, w, h))
            dataL_noc = np.ascontiguousarray(dataL_noc,dtype=np.float32)/256

            processed = preprocess.get_transform(augment=False)  
            left_img       = processed(left_img)
            right_img      = processed(right_img)

            #FLIP_VERTICALLY:
            probability = np.random.random()
            if self.flip_vertical == 1 and probability > 0.5:
                left_img = left_img.flip(-2)
                right_img = right_img.flip(-2)
                dataL = np.flip(dataL, -2)
                dataL_noc = np.flip(dataL_noc, -2)

            temp_tensor = torch.empty(1, 1)
            return left_img, right_img, dataL,dataL_noc,temp_tensor , left_path,right_path


    def __len__(self):
        return len(self.left)
