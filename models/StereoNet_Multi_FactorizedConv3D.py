import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
from .factorizer import Factorizer
from .spatioTemporalConv_General import SpatioTemporalConv
from .fpn import FPN, LayerScale
from .cv import initCV, res_and_rec_CV
#from .spatioTemporalConv import SpatioTemporalConv
###############
import time
###############
#from .pytorch_memlab import LineProfiler,profile, set_target_gpu, MemReporter
#torch.backends.cudnn.benchmark = False  
#from .memory import log_mem, log_mem_amp, log_mem_amp_cp, log_mem_cp
#from .plot import plot_mem, pp
#import pandas as pd
##############
cuda1 = torch.device('cuda:0')
repetitions=5
rep = 0
num_components = 8
comp_names={"fe":0,"cv":1,"cf":2,"st0":3,"st1":4,"st2":5,"st3":6,"ft":7}
com_fullNames=["Feature_Extraction", "Cost_Volume", "Cost_Filtering","Stage0","Stage1","Stage2","Stage3","Full_time"]
timings_components=[]
for i in range(num_components):
    timings_components.append(np.zeros((repetitions,1)))
#print(len(timings_components[0]))
##############

def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation, model_bn=1):
    
    if model_bn ==1:
        return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation>1 else pad,
            dilation=dilation),
       nn.BatchNorm2d(out_channel))
    else: # model_bn=0
        return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation>1 else pad,
            dilation=dilation))
    
def soft_argmin(cost_volume):
    """Remove single-dimensional entries from the shape of an array."""
    # cost_volume_D_squeeze = torch.squeeze(cost_volume, dim=1)

    softmax = nn.Softmax(dim=1)
    disparity_softmax = softmax(-cost_volume)

    d_grid = torch.arange(cost_volume.shape[1], dtype=torch.float)
    d_grid = d_grid.reshape(-1, 1, 1)
    d_grid = d_grid.repeat((cost_volume.shape[0], 1, cost_volume.shape[2], cost_volume.shape[3])) # [batchSize, 1, h, w]
    d_grid = d_grid.to('cuda')

    tmp = disparity_softmax*d_grid
    arg_soft_min = torch.sum(tmp, dim=1, keepdim=True)

    return arg_soft_min
        
class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.disp = torch.cuda.FloatTensor(
            np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])) #torch.Size([1, 12, 1, 1])

    def forward(self, x): # x -->torch.Size([7, 1, 34, 60])
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3]) #torch.Size([7, 12, 34, 60])
        out = torch.sum(x * disp, 1)  
        return out #torch.Size([7, 34, 60])

class CostVolumeFiltering(nn.Module):
    def __init__(self, use_skip,is_filter1_differ,filter1_kernels,fact_kernels,ch_in, ch_out, subspace_scale, stream_axes,BN_1D,BN_2D,BN_1D_last,model_bn):
        super().__init__()
        self.filterBlocks = nn.ModuleList()
        self.use_skip = use_skip
        next_ch_out =ch_out
        assert (is_filter1_differ == 1 or is_filter1_differ ==0), "is_filter1_differ argument must have value 0 or 1"
        no_streams = len(stream_axes)
        if is_filter1_differ==1:
            factorizer_filter1 = Factorizer(filter1_kernels, ch_in, ch_out, subspace_scale, stream_axes)
            for i, stream in enumerate(factorizer_filter1.streams):
                for _ in range(1):
                    self.filterBlocks.append(SpatioTemporalConv(stream,ch_in,ch_out,model_bn,False,BN_1D,BN_2D,BN_1D_last))
        ####
        factorizer = Factorizer(fact_kernels, ch_in, ch_out, subspace_scale, stream_axes)
        for i, stream in enumerate(factorizer.streams):
            for i in range(4-is_filter1_differ): #4
                if is_filter1_differ == 1 or i !=0: # 
                    ch_in = ch_out
                
                self.filterBlocks.append(SpatioTemporalConv(stream,ch_in,ch_out,model_bn,False,BN_1D,BN_2D,BN_1D_last))
            self.filterBlocks.append(SpatioTemporalConv(stream,ch_in,1,model_bn,True))
            
    def forward(self, x):
        if self.use_skip == 0:
            for f in self.filterBlocks:
                x= f(x)
            return x
        else:
            x0 = self.filterBlocks[0](x)
            x1 = self.filterBlocks[1](x0)
            x2 = self.filterBlocks[2](x0 +x1)  
            x3 = self.filterBlocks[3](x2)
            x4 = self.filterBlocks[4](x2+x3)
            return x4
#############
class CostVolume3D_Filtering(nn.Module):
    # model_bn = 1 : allow BN in your model and then we look to the value of BN_2D_last, model_bn = 0 --> don't use BN on your model
    # if BN_2D_last = 1 --> use BN in the last conv2d only. BN_2D_last = 0 --> use BN on all conv2d layers
    def __init__(self, ch_in,ch_out,kernal_size,stride, pad, BN_2D_last, model_bn , bias=False): 
        super().__init__()
        self.filterBlocks = nn.Sequential()
        for i in range(5): # number of Conv2d in each filter block
            if i == 4: # last conv2d
                self.filterBlocks.add_module("conv2d_" +str(i), nn.Conv2d(ch_in, 1, kernal_size, stride, pad))
                if model_bn == 1:
                    self.filterBlocks.add_module('BN_'+str(i), nn.BatchNorm2d(1))
                self.filterBlocks.add_module('LeakyReLU_'+str(i), nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                self.filterBlocks.add_module("conv2d_" +str(i), nn.Conv2d(ch_in, ch_out, kernal_size, stride, pad))
                if model_bn == 1:
                    if BN_2D_last == 0:
                        self.filterBlocks.add_module('BN_'+str(i), nn.BatchNorm2d(ch_out))
                self.filterBlocks.add_module('LeakyReLU_'+str(i), nn.LeakyReLU(negative_slope=0.2, inplace=True))
                    
    def forward(self, x):
        x = self.filterBlocks(x)
        return x

class StereoNet(nn.Module):
    def __init__(self, r, use_skip,initial_ch, num_convs_in_layers, initial_scale_factor, disp_offset,sub_pixel_acc,patch_index,is_costvolume_4D,BN_2D_last, chout_costfiltring, is_filter1_differ,filter1_kernels,fact_kernels,BN_1D,BN_2D,BN_1D_last,model_bn,maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.r = r
        self.initial_ch =initial_ch
        self.num_convs_in_layers=num_convs_in_layers
        self.initial_scale_factor=initial_scale_factor
        self.disp_offset=disp_offset
        self.sub_pixel_acc=sub_pixel_acc # increment disp_offset range values by 0.5 or 1
        self.is_costvolume_4D = is_costvolume_4D # is generated cost volume 4D with dimension (B,1,D,H,W) or 3D volume (B,D,H,W)
        self.chout_costfiltring = chout_costfiltring
        self.feature_extraction = FPN(initial_ch,LayerScale, num_convs_in_layers,initial_scale_factor)
        self.initcv = initCV(maxdisp)
        self.res_rec_cv = res_and_rec_CV(patch_index)
        self.filter = nn.ModuleList()
        no_feature_layers = len(num_convs_in_layers)
        # first filter in the "self.filter" list for the coarse resolution with highest number of channels
        
        if is_costvolume_4D == 0: # for 3D cost volume
            for lay_ind in range(no_feature_layers-1,-1,-1):
                if lay_ind == no_feature_layers-1:
                    ch_in = (maxdisp+1) // (initial_scale_factor * (2 ** lay_ind ))
                    ch_out = ch_in
                else:
                    if sub_pixel_acc == 1 or sub_pixel_acc == 0.5:
                        num_disp_offset_range = (((disp_offset+1)*2//sub_pixel_acc) - (1//sub_pixel_acc)) * 2
                    else:
                        num_disp_offset_range = ((disp_offset+1)*2 - 1) * 2 # we multiply by 2 because we concatentaed Residual cost and Reconstruction volume error on channel dimension
                    ch_in = int(num_disp_offset_range)
                    ch_out = ch_in
                self.filter.append(CostVolume3D_Filtering(ch_in,ch_out,3,1, 1,BN_2D_last, model_bn , bias=False))
        else: # for 4D cost volume ch_in = ch_out = 1
            for lay_ind in range(no_feature_layers-1,-1,-1):
                if lay_ind == no_feature_layers-1:
                    if chout_costfiltring <= 0: #chout_costfiltring <= 0 means ch_out is derived from initial_scale_factor value. if chout_costfiltring > 0, use this value for ch_out
                        ch_in = 1
                        ch_out = (maxdisp+1) // (initial_scale_factor * (2 ** lay_ind ))
                    else:
                        ch_in = 1 #chout_costfiltring
                        ch_out = chout_costfiltring
                else: ## for residual and reconstruction cost
                    if sub_pixel_acc == 1 or sub_pixel_acc == 0.5:
                        num_disp_offset_range = (((disp_offset+1)*2//sub_pixel_acc) - (1//sub_pixel_acc)) * 2
                    else:
                        num_disp_offset_range = ((disp_offset+1)*2 - 1) * 2 # we multiply by 2 because we concatentaed Residual cost and Reconstruction volume error on channel dimension
                    ch_in = 1 #int(num_disp_offset_range)
                    ch_out = int(num_disp_offset_range)
                self.filter.append(CostVolumeFiltering(use_skip,is_filter1_differ,filter1_kernels,fact_kernels,ch_in, ch_out, 1, [1],BN_1D,BN_2D,BN_1D_last,model_bn))
                
                '''
                if chout_costfiltring <= 0 and lay_ind == no_feature_layers-1:
                    self.filter.append(CostVolumeFiltering(is_filter1_differ,filter1_kernels,fact_kernels,ch_in, ch_out, 1, [1],BN_1D,BN_2D,BN_1D_last,model_bn))
                    ch_in=ch_out
                else:
                    self.filter.append(CostVolumeFiltering(is_filter1_differ,filter1_kernels,fact_kernels,ch_in, ch_out, 1, [1],BN_1D,BN_2D,BN_1D_last,model_bn))
                '''
            #for lay_ind in range(no_feature_layers):
            #    self.filter.append(CostVolumeFiltering(is_filter1_differ,filter1_kernels,fact_kernels,1, 1, 1, [1],BN_1D,BN_2D,BN_1D_last,model_bn))
        
    #@profile
    def forward(self, left, right):
        _,_,img_h,img_w = left.size() #torch.Size([7, 3, 540, 960])
        ## Feature extraction using FPN(Feature Pyramid Network)
        top_down_refimg_feature, scale_layer_refimg = self.feature_extraction(left) # top_down_refimg_feature =[torch.Size([7, 32, 34, 60],torch.Size([7, 16, 68, 120], torch.Size([7, 8, 135, 240]]
        top_down_targetimg_feature, scale_layer_targetimg = self.feature_extraction(right) # scale_layer_refimg = [16, 8, 4]
        #print("scale_layer_refimg: ", scale_layer_refimg)
        #for i in top_down_refimg_feature:
        #    print("top_down_refimg_feature", top_down_refimg_feature[i].size())
        #exit()
        ## initial Cost volume
        # scale index 0 contains the coarse feature map usinf a coarse scale factor
        pred_list=[]
        pred_org_size_list=[]
        for scale_ind in range(len(scale_layer_refimg)):
            f_h, f_w = top_down_refimg_feature[scale_ind].size(2), top_down_refimg_feature[scale_ind].size(3)
            #print("top_down_refimg_feature size: ", top_down_refimg_feature[scale_ind].size())
            if scale_ind == 0: # get initial cost volume on coarse scale
                cost = self.initcv(top_down_refimg_feature[scale_ind], top_down_targetimg_feature[scale_ind], scale_layer_refimg[scale_ind])
                ##cost size = torch.Size([B, 12, 34, 60])
            else: # get concatentated residual and reconstruction error cost volumes
                prev_disp_org_size= pred_org_size_list[scale_ind-1]
                prev_disp_org_size = prev_disp_org_size * f_w / prev_disp_org_size.size(-1)
                prev_disp_up = F.interpolate(prev_disp_org_size,size=(f_h, f_w),mode='bilinear', align_corners=True)  #torch.Size([7, 1, 68, 120])
                prev_disp_up = torch.squeeze(prev_disp_up,1)
                
                cost = self.res_rec_cv(top_down_refimg_feature[scale_ind], top_down_targetimg_feature[scale_ind], prev_disp_up,self.disp_offset,self.sub_pixel_acc)   #torch.Size([7, 10, 68, 120])   
            
            ### cost filtering using factorizing
            #print("cost size: ", cost.size())
            if self.is_costvolume_4D == 1:
                cost = torch.unsqueeze(cost, 1)
                cost = self.filter[scale_ind](cost) 
                cost = torch.squeeze(cost, 1) 
                pred =soft_argmin(cost) 
                
            else: # cost volume 3d
                cost = self.filter[scale_ind](cost)
                pred =soft_argmin(cost)

            if scale_ind==0: # for initial volume
                pred_low_res = disparityregression((self.maxdisp + 1) // scale_layer_refimg[scale_ind])(pred) 
                pred_low_res= torch.unsqueeze(pred_low_res,1) 
                pred_h, pred_w = pred_low_res.size(-2),pred_low_res.size(-1)
                pred_org_size_list.append(pred_low_res)

                pred_low_res = pred_low_res * img_w / pred_w
                disp_up = F.interpolate(pred_low_res,size=(img_h, img_w),mode='bilinear', align_corners=True)
                pred_list.append(disp_up)
            else:
                # residual disparity
                pred_low_res = disparityregression(self.disp_offset)(pred)
                pred_low_res= torch.unsqueeze(pred_low_res,1)
                pred_h, pred_w = pred_low_res.size(-2),pred_low_res.size(-1)
                sum_res_disp = pred_low_res + torch.unsqueeze(prev_disp_up,1)
                pred_org_size_list.append(sum_res_disp)

                sum_res_disp = sum_res_disp * img_w / pred_w
                disp_up = F.interpolate(sum_res_disp,size=(img_h, img_w),mode='bilinear', align_corners=True) 
                pred_list.append(disp_up)
                
                #pred_low_res = pred_low_res * img_w / pred_w
                #disp_up = F.interpolate(pred_low_res,size=(img_h, img_w),mode='bilinear', align_corners=True) 
                # add residiual disparity to the previous disparity map
                #pred_list.append(disp_up+pred_list[scale_ind-1])
        '''
        for p in pred_list:
            print("pred_list", p.size())
        print("#########")
        for p in pred_org_size_list:
            print("pred_org_size_list", p.size())
        exit()
        '''
        return pred_list
        #return pred_list , pred_org_size_list
