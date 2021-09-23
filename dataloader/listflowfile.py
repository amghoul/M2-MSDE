import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename): 
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, data_range_train= 200, data_range_Val=38): # 84% /home/alghoul/myenv/FlyingThings3D
    # flyingthings
    # training dataset path.
    filename_left     = filepath+ r"/train/image_clean/left/"
    filename_right    = filepath+ r"/train/image_clean/right/"
    filename_disp_L = filepath+ r"/train/disparity/left/"
    filename_disp_R = filepath+ r"/train/disparity/right/"
    filename_disp_L_OCC = filepath+ r"/train/disparity_occlusions/left/"
    
    # The dataset will be loaded into data and label numpy arrays. Some manipulation might be needed.
    train_images_left  = [filename_left + '00{:05d}.png'.format(id) for id in range(data_range_train)]
    train_images_right = [filename_right + '00{:05d}.png'.format(id) for id in range(data_range_train)]
    train_gt_disparity_L = [filename_disp_L + '00{:05d}.pfm'.format(id) for id in range(data_range_train)]
    train_gt_disparity_R = [filename_disp_R + '00{:05d}.pfm'.format(id) for id in range(data_range_train)] 
    train_gt_disparity_L_OCC = [filename_disp_L_OCC + '00{:05d}.png'.format(id) for id in range(data_range_train)]

    # test
    filename_left     = filepath+ r"/val/image_clean/left/"
    filename_right    = filepath+ r"/val/image_clean/right/"
    filename_disp_L = filepath+ r"/val/disparity/left/"
    filename_disp_R = filepath+ r"/val/disparity/right/"
    filename_disp_L_OCC = filepath+ r"/val/disparity_occlusions/left/"

    # The dataset will be loaded into data and label numpy arrays. Some manipulation might be needed.
    test_images_left  = [filename_left + '00{:05d}.png'.format(id) for id in range(data_range_Val)]
    test_images_right = [filename_right + '00{:05d}.png'.format(id) for id in range(data_range_Val)]
    test_gt_disparity_L = [filename_disp_L + '00{:05d}.pfm'.format(id) for id in range(data_range_Val)]
    test_gt_disparity_R = [filename_disp_R + '00{:05d}.pfm'.format(id) for id in range(data_range_Val)] 
    test_gt_disparity_L_OCC = [filename_disp_L_OCC + '00{:05d}.png'.format(id) for id in range(data_range_Val)]
    
    return train_images_left, train_images_right, train_gt_disparity_L,train_gt_disparity_L_OCC, test_images_left, test_images_right, test_gt_disparity_L,test_gt_disparity_L_OCC