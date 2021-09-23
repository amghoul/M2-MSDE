import torch
import os
import torch.utils.data
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

#from utils.save_load import load_dataset
class transformed_data(Dataset):
  def __init__(self, img, transform=None):
    self.img = img  #img path
    self.len = len(os.listdir(self.img))

  def __getitem__(self, index):
    ls_img = sorted(os.listdir(self.img))
    img_file_path = os.path.join(self.img, ls_img[index])
    loaded = np.load('{}'.format(img_file_path))
    left_img = torch.from_numpy(loaded['l'])
    right_img = torch.from_numpy(loaded['r'])
    disp_L = torch.from_numpy(loaded['dl'])
    disp_L_occ_noc = torch.from_numpy(loaded['dlon'])
    obj_map_img = torch.from_numpy(loaded['omi'])
    left_path = int(loaded['lp'])
    right_path = int(loaded['rp'])
    return left_img, right_img, disp_L, disp_L_occ_noc, obj_map_img , left_path,right_path

  def __len__(self):
    return self.len  

def load_dataset2(datapath,dataset,datatype,flip_vertical = False):
    #dataset = "sceneflow"
    #datatype = "2015"
    train_bsize = 1
    test_bsize = 1
    #flip_vertical = False
    if dataset == "kitti":
        if datatype == '2015':
            from dataloader import KITTIloader2015 as ls
            from dataloader import KITTILoader as DA
            train_left_img, train_right_img, train_left_disp,train_left_disp_noc, test_left_img, test_right_img, test_left_disp,test_left_disp_noc,train_mask_obj_map,test_mask_obj_map = ls.dataloader(
            datapath)
            
        else:# datatype == '2012':
            from dataloader import KITTIloader2012 as ls
            from dataloader import KITTILoader1 as DA
            train_left_img, train_right_img, train_left_disp,train_left_disp_noc, test_left_img, test_right_img, test_left_disp,test_left_disp_noc = ls.dataloader(
            datapath)
            
    else: ##sceneflow dataset
        from dataloader import listflowfile as lt  ## change import fie for scenflow dataset
        #from dataloader import SecenFlowLoader1 as DA
        from dataloader import SecenFlowLoaderMy as DA
        
        train_left_img, train_right_img, train_left_disp, train_left_disp_occ ,test_left_img, test_right_img, test_left_disp, test_left_disp_occ= lt.dataloader(
            datapath,data_range_train= 21818,data_range_Val=4248) #21818 4248
        '''
        train_left_img, train_right_img, train_left_disp, train_left_disp_occ ,test_left_img, test_right_img, test_left_disp, test_left_disp_occ = lt.dataloader(
            datapath,data_range_train= 7,data_range_Val=3) #200 4248
        '''
    train_left_img.sort()
    train_right_img.sort()
    train_left_disp.sort()
    if dataset == "kitti":
        train_left_disp_noc.sort()
        if datatype == '2015': 
            train_mask_obj_map.sort()
    else:
        train_left_disp_occ.sort()
    
    test_left_img.sort()
    test_right_img.sort()
    test_left_disp.sort()
    if dataset == "kitti":
        test_left_disp_noc.sort()
        if datatype == '2015': 
            test_mask_obj_map.sort()
    else:
        test_left_disp_occ.sort()
    
    #__normalize = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    __normalize = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
    if dataset == "kitti":
        if datatype == '2015':
            DA.myImageFloder(train_left_img, train_right_img, train_left_disp,train_left_disp_noc,train_mask_obj_map, True,flip_vertical)
            TrainImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(train_left_img, train_right_img, train_left_disp,train_left_disp_noc,train_mask_obj_map, True,flip_vertical),
                batch_size=train_bsize, shuffle=False, num_workers=12, drop_last=False) ## org shuffle = False
            
            #else: #mode= test
            TestImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(test_left_img, test_right_img, test_left_disp,test_left_disp_noc,test_mask_obj_map, False,flip_vertical),
                batch_size=test_bsize, shuffle=False, num_workers=4, drop_last=False)
        else: #datatype == '2012':
            TrainImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(train_left_img, train_right_img, train_left_disp,train_left_disp_noc,True,flip_vertical),
                batch_size=train_bsize, shuffle=False, num_workers=12, drop_last=False) ## org shuffle = False
            #else: #mode= test
            TestImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(test_left_img, test_right_img, test_left_disp,test_left_disp_noc,False,flip_vertical),
                batch_size=test_bsize, shuffle=False, num_workers=4, drop_last=False)

    else: # dataset== "sceneflow"
        #if args.mode == "train":
        TrainImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(train_left_img, train_right_img, train_left_disp,train_left_disp_occ, True, normalize=__normalize),
            batch_size=train_bsize, shuffle=False, num_workers=12, drop_last=False)
        #else: #mode= test
        TestImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(test_left_img, test_right_img, test_left_disp,test_left_disp_occ, False, normalize=__normalize),
            batch_size=test_bsize, shuffle=False, num_workers=4, drop_last=False)#4
    return TrainImgLoader,TestImgLoader

def save_dataset_numpy(org_root_path_dataset,new_save_filepath,dataset,datatype,flip_vertical):
        if dataset == "sceneflow":
            if os.path.isdir(new_save_filepath) == False:
                os.makedirs(new_save_filepath)
            filename_train     = new_save_filepath+ r"/FlyingThings3D/train/"
            filename_val    = new_save_filepath+ r"/FlyingThings3D/val/"
            if os.path.isdir(filename_train) == False:
                os.makedirs(filename_train)
            if os.path.isdir(filename_val) == False:
                os.makedirs(filename_val)
        else:
            if datatype == "2015":
                new_save_filepath = (new_save_filepath+"/kitti/2015")
            else:
                new_save_filepath = (new_save_filepath+"/kitti/2012")
            
            if os.path.isdir(new_save_filepath) == False:
                os.makedirs(new_save_filepath)
            filename_train     = new_save_filepath+ r"/train/"
            filename_val    = new_save_filepath+ r"/val/"
            if os.path.isdir(filename_train) == False:
                os.makedirs(filename_train)
            if os.path.isdir(filename_val) == False:
                os.makedirs(filename_val)

        ## load and preprocess on original datraset
        TrainImgLoader, TestImgLoader = load_dataset2(org_root_path_dataset,dataset,datatype,flip_vertical = False)
        print("TrainImgLoader length: ", len(TrainImgLoader))
        print("TestImgLoader length: ", len(TestImgLoader))
        
        ## save the preprocessed dataset to numpy compressed
        save_train_val_to_numpy(TestImgLoader,filename_val )
        print("Finished saving validation dataset to numpy in hardisk in {}".format(filename_val))
        save_train_val_to_numpy(TrainImgLoader,filename_train )
        print("Finished saving train dataset to numpy in hardisk in {}".format(filename_train))

def save_train_val_to_numpy(imgLoader,save_path):
    for batch_idx, (imgL, imgR, disp_L, disp_L_occ_noc, obj_map_img,left_path,right_path) in enumerate(imgLoader):
        image_name = left_path[0].split("/")[-1].split(".")[0]
        imgL = torch.squeeze(imgL,0)
        imgR = torch.squeeze(imgR,0)
        disp_L = torch.squeeze(disp_L,0)
        disp_L_occ_noc = torch.squeeze(disp_L_occ_noc,0)
        obj_map_img = torch.squeeze(obj_map_img,0)
        left_path = left_path[0]
        right_path = right_path[0]
        np.savez_compressed('{}{}.npz'.format(save_path,image_name), l = imgL, r = imgR, dl = disp_L, dlon = disp_L_occ_noc, omi = obj_map_img, lp= image_name,rp = image_name )
        print("saved train image {}/{} ".format(image_name,len(imgLoader)))

def load_saved_numpy_dataset(filename,batch_size,shuffle,num_workers):
    loadedDataset = transformed_data(filename)
    print("The dataset length is: " , loadedDataset.__len__())
    new_ImgLoader=DataLoader(loadedDataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return new_ImgLoader
    
if __name__ == '__main__':

    org_root_path_dataset = "/ds-av/public_datasets/freiburg_sceneflow_subset/raw"
    new_save_filepath = ("/netscratch/alghoul/myDataset")
    dataset = "sceneflow"
    datatype = "2015"
    flip_vertical = False

    save_dataset_numpy(org_root_path_dataset,new_save_filepath,dataset,datatype,flip_vertical=False)
    '''
    batch_size = 2
    if dataset == "sceneflow":
        filename = new_save_filepath + "/FlyingThings3D"
        train_img_loader = load_saved_numpy_dataset(filename+"/train/",batch_size,shuffle = True,num_workers=12)
        test_img_loader = load_saved_numpy_dataset(filename+"/val/",batch_size,shuffle = False,num_workers=4)
    '''