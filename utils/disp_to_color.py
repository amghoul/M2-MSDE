import glob
import torch
import cv2 as cv
import os
import numpy as np
import torch.nn.functional as F
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

def disp_write( disparity, args):
    I = disparity * 256.0
    if args.dataset == "kitti":
        I[disparity==0] = 1
        I[I < 0 ] = 0
        I[I>65535] = 0
    return I

def disp_map(I,disp):
    map = [[0,0,0,114], [0,0,1,185], [1,0,0,114], [1,0,1,174],
        [0,1,0,114], [0,1,1,185],[1,1,0,114], [1,1,1,0]]
    map = np.array(map,dtype=np.int)
    bins = map[:-1,3]
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]
    ind = sum(np.tile(I,[6,1]) > np.tile(cbins,[I.size,1]).T)
    bins  = 1 / bins
    cbins = np.insert(cbins,0,0.0)

    t1 = [cbins[t] for t in ind]
    t1 = np.array(t1)
    t2 = [bins[t] for t in ind]
    t2 = np.array(t2)
    I = (I-t1) * t2
    #map(ind+1,1:3) .* repmat(I, [1 3])
    t3 = [map[t,:3] for t in ind]
    t3 = np.array(t3)
    t3 = t3 * np.tile(1-I.T,[3,1]).T
    t4 = [map[t+1,:3] for t in ind]
    t4 = np.array(t4)
    t4 = t4 * np.tile(I.T,[3,1]).T
    t5 = t3 + t4
    t5[np.where(t5<0)] = 0
    t5[np.where(t5>1)] = 1
    t5 = np.reshape(t5,[disp.shape[0],disp.shape[1],3],order='F') * 255
    t5 = np.array(t5,dtype=np.uint8)
    t5 = cv.cvtColor(t5,cv.COLOR_RGB2BGR)
    return t5

def save_colored_disp(disp, dest,max_disp=65):
    #max_disp = 192
    #output_disp_dir = "/home/alghoul/myenv/All-StereoNets/StereoNet-Last-forMe-V18/utils/"
    #disp = "/home/alghoul/myenv/All-StereoNets/StereoNet-Last-forMe-V18/utils/000086_10.png"
    #print('colorlize ',disp)
    #img = cv2.imread(disp,-1)
    #print(img.shape)
    #I = np.uint16(np.around(img/256))
    #I = np.array(I,dtype=np.float)
    I = (disp.T.flatten()) / max_disp
    r = disp_map(I,disp)
    cv.imwrite(dest, r)

def save_images(outputs,disp_L,left_path,right_path,batch_idx,ABS_THRESH,args,test_results_path):
    '''
    turbo_colormap_data = get_color_map()
    mpl_data = RGBToPyCmap(turbo_colormap_data)
    plt.register_cmap(name='turbo', data=mpl_data, lut=turbo_colormap_data.shape[0])
    '''
    stages = args.stages
    save_path_stages = test_results_path + '/' + "disp_map_stages"
    save_path_combined = test_results_path + '/' + "disp_map_combined" 
    save_path_colors = test_results_path + '/' + "colors_disp_map"
    
    #### create folders if not exist
    CHECK_FOLDER = os.path.isdir(save_path_stages)
    if not CHECK_FOLDER:
        os.makedirs(save_path_stages)
    CHECK_FOLDER = os.path.isdir(save_path_combined)
    if not CHECK_FOLDER:
        os.makedirs(save_path_combined)
    CHECK_FOLDER = os.path.isdir(save_path_colors)
    if not CHECK_FOLDER:
        os.makedirs(save_path_colors)

    _, H, W = disp_L.size()
    #_, H, W = outputs[0].shape
    all_results = torch.zeros((len(outputs)+1, 1, H, W))
    org_name= left_path[0].split('/')[-1].split('.')[0]
    org_imgL = cv.imread(left_path[0])
    org_imgR = cv.imread(right_path[0])
    ### save original left and right images
    cv.imwrite(join(save_path_colors, args.mode +'-'+args.dataset+'-'+args.datatype+'-imgL-%s.png' % (org_name)),org_imgL)
    cv.imwrite(join(save_path_colors, args.mode +'-'+args.dataset+'-'+args.datatype+'-imgR-%s.png' % (org_name)),org_imgR)
    #concat_output=[]
    for j in range(len(outputs)):
        output_size = outputs[j].size()
        if len(output_size) == 3:
            _, _, out_w = outputs[j].size()
        elif len(output_size) == 4:
            _,_, _, out_w = outputs[j].size()
            if out_w != W:
                outputs[j] = F.interpolate(outputs[j],size=(H, W),mode='bilinear', align_corners=True)
            outputs[j] = outputs[j].squeeze(1)
        else:
            print("Nuumber of dimensions in the output is larger than 4, it must be 3 or 4.")
            exit()
            
    for j in range(len(outputs)):
        disp_save = disp_write(outputs[j][0, :, :],args)
        disp_save_16bits = np.array(disp_save.detach().cpu().numpy(), dtype=np.uint16)
        #concat_output.append(disp_save_16bits)
        all_results[j, 0, :, :] = outputs[j][0, :, :]/256.0
        ###############
        image_name_to_save= join(save_path_stages, args.mode +'-'+args.dataset+'-'+args.datatype+'-iter-%d-stage-%d-OrgName-%s.png' % (batch_idx,j,org_name))
        cv.imwrite(image_name_to_save, ( np.array(disp_write(outputs[j][0, :, :],args).detach().cpu().numpy(), dtype=np.uint16)))
        
    ############## Saving Ground truth
    disp_save = disp_write(disp_L[0,:, :],args)
    disp_save_16bits = np.array(disp_save.detach().cpu().numpy(), dtype=np.uint16)
    #concat_output.append(disp_save_16bits)

    image_name_to_save = join(save_path_stages, args.mode +'-'+args.dataset+'-'+args.datatype+'-iter-%d-OrgName-%s-GT.png' % (batch_idx,org_name))
    cv.imwrite(image_name_to_save, ( np.array(disp_write(disp_L[0,:, :],args).detach().cpu().numpy(), dtype=np.uint16)))
    all_results[-1, 0, :, :] = disp_L[0,:, :]/256.0
    
    image_name_to_save = join(save_path_combined, args.mode +'-'+args.dataset+'-'+args.datatype+'-iter-%d-OrgName-%s.png' % (batch_idx,org_name))
    '''
    im_h = cv.hconcat(concat_output)
    cv.imwrite(image_name_to_save, im_h)
    '''
    ########## save disp maps as colored images
    _, H, W = outputs[0].shape
    all_results_color = torch.zeros((H, (stages+1)*W))
    for j in range(len(outputs)):
        all_results_color[:,j*W:(j+1)*W]= outputs[j][0, :, :]

    all_results_color[:,len(outputs)*W:(len(outputs)+1)*W]= disp_L[0,:, :]
    disp_save_16bits = np.array(all_results_color.detach().cpu().numpy(), dtype=np.uint16)

    #### save combined uncolored disparity
    cv.imwrite(image_name_to_save, ( np.array(disp_write(all_results_color,args).detach().cpu().numpy(), dtype=np.uint16)))
    ####save combined colored disparity
    image_name_to_save = join(save_path_colors, args.mode +'-'+args.dataset+'-'+args.datatype+'-iterpredcolor-%d-OrgName-%s.png' % (batch_idx,org_name))
    image_name_to_save_jet = join(save_path_colors, args.mode +'-'+args.dataset+'-'+args.datatype+'-iterpredcolor-%d-OrgName-JET-%s.png' % (batch_idx,org_name))
    if args.dataset == "kitti":
        save_colored_disp(disp_save_16bits,image_name_to_save, 68) #65
        im_color = cv.applyColorMap(np.array(all_results_color.detach().numpy()*2, dtype=np.uint8), cv.COLORMAP_JET)
        cv.imwrite(image_name_to_save_jet,im_color)
    else:
        save_colored_disp(disp_save_16bits,image_name_to_save, 110) #103
        im_color = cv.applyColorMap(np.array(all_results_color.detach().numpy()*2, dtype=np.uint8), cv.COLORMAP_JET)
        cv.imwrite(image_name_to_save_jet,im_color)
    '''
    image_name_to_save = join(save_path_colors, args.mode +'-'+args.dataset+'-'+args.datatype+'-iterpredcolor-%d-OrgName-colormap-%s.png' % (batch_idx,org_name))
    plot_disparity(image_name_to_save, all_results_color, 68)#192
    ########3
    image_name_to_save = join(save_path_colors, args.mode +'-'+args.dataset+'-'+args.datatype+'-iterpredcolor-%d-OrgName-RAINBOW-%s.png' % (batch_idx,org_name))
    im_color = cv.applyColorMap(np.array(all_results_color.detach().numpy()*2, dtype=np.uint8), cv.COLORMAP_RAINBOW)
    cv.imwrite(image_name_to_save,im_color)

    image_name_to_save = join(save_path_colors, args.mode +'-'+args.dataset+'-'+args.datatype+'-iterpredcolor-%d-OrgName-JET-%s.png' % (batch_idx,org_name))
    im_color = cv.applyColorMap(np.array(all_results_color.detach().numpy()*2, dtype=np.uint8), cv.COLORMAP_JET)
    cv.imwrite(image_name_to_save,im_color)
    
    image_name_to_save = join(save_path_colors, args.mode +'-'+args.dataset+'-'+args.datatype+'-iterpredcolor-%d-OrgName-HSV-%s.png' % (batch_idx,org_name))
    im_color = cv.applyColorMap(np.array(all_results_color.detach().numpy()*2, dtype=np.uint8), cv.COLORMAP_HSV)
    cv.imwrite(join(
        save_path_colors, args.mode +'-'+args.dataset+'-'+args.datatype+'-iterpredcolor-%d-OrgName-HSV-%s.png' % (batch_idx,org_name)),im_color)
    '''


