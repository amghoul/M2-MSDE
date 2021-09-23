import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  left_fold  = '/image_2/'
  right_fold = '/image_3/'
  disp_L = '/disp_occ_0/' #Ground Truth
  disp_R = '/disp_occ_1/'
  disp_L_noc = '/disp_noc_0/'
  mask_obj_map = '/obj_map/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

  train = image[:16]#image[:160]
  
  val   = image[160:]#[160:]

  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train_L = [filepath+disp_L+img for img in train]
  disp_train_L_noc = [filepath+disp_L_noc+img for img in train]
  #disp_train_R = [filepath+disp_R+img for img in train]
  mask_obj_map_train = [filepath+mask_obj_map+img for img in train]
  #print(mask_obj_map_train)
  #exit()
  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val_L = [filepath+disp_L+img for img in val]
  disp_val_L_noc = [filepath+disp_L_noc+img for img in val]
  #disp_val_R = [filepath+disp_R+img for img in val]
  mask_obj_map_val = [filepath+mask_obj_map+img for img in val]
  #print(left_val)
  #print(mask_obj_map_val)
  #exit()
  return left_train, right_train, disp_train_L, disp_train_L_noc, left_val, right_val, disp_val_L,disp_val_L_noc, mask_obj_map_train, mask_obj_map_val
