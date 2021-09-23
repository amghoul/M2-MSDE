########################################
import torch
import torch.nn as nn
# this class calculate initial cost volume at coarse feature map scale
class initCV(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.maxdisp = maxdisp
        
    def forward(self,coarse_fm_left, coarse_fm_right, coarse_fm_scale):
        # coarse_fm_left: coarse feature map for the left imae input
        # coarse_fm_scale: coarse feature scale for the coarse feature map
        coarse_disp = (self.maxdisp + 1) // coarse_fm_scale

        cost = torch.cuda.FloatTensor(coarse_fm_left.size()[0],coarse_disp,coarse_fm_left.size()[2],coarse_fm_left.size()[3]).zero_()
        #print(cost.size())
        for i in range(coarse_disp):
            cost[:, i, :, :i] = torch.sum(torch.abs(coarse_fm_left[:, :, :, :i]),1)
            if i > 0:
                cost[:, i, :,i:] = torch.sum(torch.abs(coarse_fm_left[ :, :, :, i:] - coarse_fm_right[:, :, :, :-i]),1) ## using l1 norm to create combined cost volume with dimension D,H,W
            else:
                cost[:, i, :, :] = torch.sum(torch.abs(coarse_fm_left - coarse_fm_right),1)
        return cost.contiguous()

##########################################
# this class calculate residual and reconstruction error cost volume from the upsampled coarse initial disparity map
class res_and_rec_CV(nn.Module):
    def __init__(self,patch_index): # where patch_index is the k value whick patch size = 2k+1
        super().__init__()
        self.patch_index = patch_index
        
    def batch_correlation_notgood(self, FeatureL,FeatureR,k=2):
        # patch size = 2k +1
        K = 2 * k + 1 # patch size
        B, C,H,W = FeatureL.size()
        output= torch.zeros_like(FeatureL).cuda()
        for h in range(H):
            for w in range(W):
                sump = torch.zeros(B,C,1,1).cuda()
                start_h = h-k # start_h: the start of the patch size in height direction
                end_h = h+k+1
                start_w = w-k
                end_w = w+k+1
                if start_h < 0:
                    start_h = 0
                if end_h > H-1:
                    end_h = H-1
                if start_w < 0:
                    start_w = 0
                if end_w > W-1:
                    end_w = W-1
                for ph in range(start_h,end_h,1): #ph: patch height
                    for pw in range(start_w,end_w,1):
                        sump += (FeatureL[:,:,ph,pw] * FeatureR[:,:,ph,pw]).view(B,C,1,1)
                output[:,:,h,w]= sump.squeeze(2).squeeze(2)
        return output
    
    def warp_old(self, x, disp_map):
        """
        warp an image/tensor (im2) back to im1
        x: [B, C, H, W] (im2)
        disp_map: initial disparity map
        """
        B, C, H, W = x.size()
        # mesh grid
        # Original coordinates of pixels
        xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = torch.cat((xx, yy), 1).float()

        # Apply shift in X direction
        # vgrid = Variable(grid)
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp_map

        # In grid_sample coordinates are assumed to be between -1 and 1
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        # vgrid dimwnsion --> B,2,H,W
        vgrid = vgrid.permute(0, 2, 3, 1) #--> vgrid becomes B,H,W,2
        warped_imag = nn.functional.grid_sample(x, vgrid) # warped_imag dimesntion--> (B,3,H,W)
        return warped_imag

    ###############################
    ## convert image to column according to the path size 2*k+1
    def im2col(self, image, k, stride_tuple=1, in_groups=1):
        in_b, in_c, in_h, in_w = image.shape
        K = 2*k+1
        k_h = K #kernel.shape[2]
        k_w = K #kernel.shape[3]
        s_h = 1 #stride_tuple[0]
        s_w = 1 #stride_tuple[1]
        #
        out_h = (in_h - k_h)//s_h + 1
        out_w = (in_w - k_w)//s_w + 1
        a = image.unfold(2, k_h, s_h)
        b = a.unfold(3, k_w, s_w)
        return b.reshape([in_b,in_groups,1,in_c//in_groups,out_h*out_w,k_h*k_w])

    def correlation(self, FeatureL,FeatureR,k=1):
        # patch size = 2k +1
        K = 2 * k + 1 # patch size
        B, C,H,W = FeatureL.size()
        m = torch.nn.ZeroPad2d(2)
        FeatureL = m(FeatureL)
        FeatureR = m(FeatureR)
        output = self.im2col(FeatureL, 2) * self.im2col(FeatureR, 2)
        output = output.sum(dim=-1).reshape(B,C,H,W)
        return output
    
    ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! be caution using - disp with kitti dataset
    def warp_with_intensities(x, y, disp_map):
        #x--> right, y --> left
        ## wraping with intensities 
        #(i',j') = T(i,j)
        #I'(i',j') = I(i,j)
        #x = right
        #y = left --> we use y intensites 
        B, C, H, W = x.size()
        ###### mesh grid
        grid_H = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        grid_V = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, 1, H, W)
        grid = torch.cat([grid_H, grid_V], 1)
        vgrid = grid.float().requires_grad_(False).cuda()
        disp_normalized = disp_map * 2 / max(W - 1, 1)
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp_normalized
        vgrid = vgrid.permute(0, 2, 3, 1) #--> vgrid becomes B,H,W,2
        y_warp = nn.functional.grid_sample(y, vgrid , padding_mode="zeros", align_corners=True) # warped_imag dimesntion--> (B,3,H,W)

        mask = torch.ones(y.size(), requires_grad=False).cuda()
        mask = nn.functional.grid_sample(mask, vgrid , padding_mode="zeros", align_corners=True)
        mask = (mask >= 1.0).float()

        return y_warp * mask
    
    def warp_without_intensities(self, x, disp_map):
        #x--> right
        #(i',j') = T(i,j)
        B, C, H, W = x.size()
        ###### mesh grid
        grid_H = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        grid_V = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, 1, H, W)
        grid = torch.cat([grid_H, grid_V], 1)
        vgrid = grid.float().requires_grad_(False).cuda()
        disp_normalized = disp_map * 2 / max(W - 1, 1)
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp_normalized
        vgrid = vgrid.permute(0, 2, 3, 1) #--> vgrid becomes B,H,W,2
        x_warp = nn.functional.grid_sample(x, vgrid , padding_mode="zeros", align_corners=True) # warped_imag dimesntion--> (B,3,H,W)

        mask = torch.ones(x.size(), requires_grad=False).cuda()
        mask = nn.functional.grid_sample(mask, vgrid , padding_mode="zeros", align_corners=True)
        mask = (mask >= 1.0).float()

        return x_warp * mask ## new cooardinates in the right image


    def forward(self,fm_left, fm_right, disp_map,disp_offset,sub_pixel_acc=1): # disp_map is the upsampled one
        #sub_pixel_acc: disp_offset increment by either 1 or 0.5
        #num_disp_offset_range: number of values in this range [-disp_offset, disp_offset]
        #print("############# before res_and_rec_CV:")
        #print("fm_left: ",fm_left.size(), "fm_right: ", fm_right.size(),"disp_map: ", disp_map.size(),"disp_offset: ",disp_offset)
        if sub_pixel_acc == 1 or sub_pixel_acc == 0.5:
            num_disp_offset_range = ((disp_offset+1)*2//sub_pixel_acc) - (1//sub_pixel_acc)
        else:
            num_disp_offset_range = (disp_offset+1)*2 - 1

        size = fm_left.size()
        #new diaprity values are [-disp_offset,disp_offset], for example if disp_offset=3 --> [-2,2] 
        # repeating disp_map across batch dimension according to the disp_offst range
        disp_map = torch.unsqueeze(disp_map,1)
        batch_disp = disp_map[:,None,:,:,:].repeat(1, int(num_disp_offset_range), 1, 1, 1).view(-1,1,size[-2], size[-1])
        #print("batch_disp before shift: ", batch_disp.size())
        # repeating disp_offset range values which it respreset (delta d)
        batch_shift = torch.arange(-disp_offset, disp_offset+1,sub_pixel_acc, device='cuda').repeat(size[0])[:,None,None,None]
        # find d_initial - delta_d --> to find the actual corrspondence point in the right image 
        batch_disp = batch_disp - batch_shift.float()
        ## repeat left and right features
        batch_fm_left = fm_left[:,None,:,:,:].repeat(1,int(num_disp_offset_range), 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        batch_fm_right = fm_right[:,None,:,:,:].repeat(1,int(num_disp_offset_range), 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        ##### residual cost
        #res_cost = torch.cuda.FloatTensor(size[0],num_disp_offset_range,size[2],size[3]).zero_()
        #print(cost.size())
        #res_cost = torch.sum(torch.abs(batch_fm_left - self.warp_without_intensities(batch_fm_right,batch_disp)),1)#.view(size[0],-1,size[-2], size[-1])
        res_cost = torch.sum(torch.abs(self.correlation(batch_fm_left, self.warp_without_intensities(batch_fm_right,batch_disp),self.patch_index)),1).view(size[0],-1,size[-2], size[-1])
        
        #########reconstruction error volume 
        
        rec_error_cost = torch.sum(torch.abs(batch_fm_left - self.warp_without_intensities(batch_fm_right, batch_disp)),1)
        rec_error_cost = rec_error_cost.view(size[0],-1, size[2],size[3])
        #print("############# after res_and_rec_CV:")
        #print("disp_map unseq: ",disp_map.size(), "batch_shift" ,batch_shift.size(),"batch_disp after shift: ", batch_disp.size())
        #print("batch_fm_left: ", batch_fm_left.size(), "batch_fm_right: ", batch_fm_right.size())
        #print("res_cost: ",res_cost.size(), "rec_error_cost: ", rec_error_cost.size())
        cost = torch.cat((res_cost,rec_error_cost),1)
        #print("cost after concatenation: ", cost.size())
        return cost.contiguous() #--> dimension B,D,H,W --> D = num_disp_offset_range

##########################################
###########################################
if __name__ == '__main__':

    from torch.autograd import Variable
    def test1():
        net = initCV(maxdisp=192)
        input = torch.randn(1,3,4,12)
        cost = net(Variable(input),Variable(input),16)
        #print("input size:", input.size())
        
        print("cost size",cost.size())
        #print(cost)
        '''
        from torchsummary import summary
        summary(net, ((3,20,30)))
        '''
    test1()

    ######################################
    from torch.autograd import Variable
    def test2():
        net = res_and_rec_CV()
        f_l = torch.randn(1,3,4,6).cuda()
        f_r = torch.randn(1,3,4,6).cuda()
        disp_map = torch.randn(1,1,4,6).cuda()
        concatinated_cost = net(Variable(f_l),Variable(f_r),Variable(disp_map),3)
        #print("input size:", input.size())
        
        print("concatinated_cost",concatinated_cost,concatinated_cost.size())
        #print("rec_error_cost",rec_error_cost,rec_error_cost.size())
        #print(cost)
        '''
        from torchsummary import summary
        summary(net, ((3,20,30)))
        '''
    test2()