import torch.nn as nn
import torch.nn.functional as F
## assume stream =1 and last layer is conv1d   

class SpatioTemporalConv(nn.Module):
    def __init__(self, stream,ch_in,ch_out,model_bn,islastBlock=False, BN_1D=1, BN_2D=0,BN_1D_last=1, is_deConv=0, bias=False): 
        #BN_1D BN_2D values: 1 or 0 i.e: use BN with conv2D or Conv1D
        #BN_1D_last {1,0} --> 1: use BN in the last layer only regardless of the BN_1D and BN_2D values. 0: use the BN_1D and BN_2D values 
        super().__init__()
        no_subsbaces= len(stream.kernels)
        stream_out_chs=[]
        stream_kernel_sizes=[]
        stream_paddings=[]
        stream_strides=[]
        stream_layers=[]
        self.is_deConv = is_deConv
        for j in range( len(stream.kernels)):
            stream_kernel_sizes.append(stream.kernels[j])
            stream_out_chs.append(stream.filters[j])
            stream_paddings.append(stream.padding[j])
            stream_strides.append(stream.strides[j])
            stream_layers.append(stream.stream_layers[j])
            #print( "subspace:",j," kernel:",stream.kernels[j], " padding:",stream.padding[j]," stride:", stream.strides[j], "out_chs:",stream.filters[j] )
        
        self.spat_temp_conv = nn.Sequential()
        if islastBlock == False: # use BN with last block
            if model_bn == 1: ## enable BN according to varaibles BN_1D & BN_2D values
                if BN_1D_last == 0:
                    for subs in range(len(stream.kernels)):
                        if subs == 0 or subs ==len(stream.kernels)-1 : # first or last conv
                            if subs == 0: # first conv
                                if self.is_deConv == 0:
                                    self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(ch_in, stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                                    stream_strides[subs], stream_paddings[subs]))
                                else:
                                    self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(ch_in, stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                                    stream_strides[subs], stream_paddings[subs]))
                                if stream_layers[subs]=="conv1d":
                                    if BN_1D ==1:
                                        self.spat_temp_conv.add_module('BN_subs'+str(subs), nn.BatchNorm3d(stream_out_chs[subs]))
                                else:
                                    if BN_2D ==1:
                                        self.spat_temp_conv.add_module('BN_subs'+str(subs), nn.BatchNorm3d(stream_out_chs[subs]))
                                #self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))
                                
                            else: # last conv1d
                                if self.is_deConv == 0:
                                    self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(stream_out_chs[subs-1], ch_out, stream_kernel_sizes[subs], 
                                                                    stream_strides[subs], stream_paddings[subs]))
                                else:
                                    self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(stream_out_chs[subs-1], ch_out, stream_kernel_sizes[subs], 
                                                                    stream_strides[subs], stream_paddings[subs]))
                                if stream_layers[subs]=="conv1d":
                                    if BN_1D ==1:
                                        self.spat_temp_conv.add_module('BN_subs'+str(subs),nn.BatchNorm3d(ch_out))
                                else:
                                    if BN_2D ==1:
                                        self.spat_temp_conv.add_module('BN_subs'+str(subs),nn.BatchNorm3d(ch_out))
                                self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))
                        
                        else: # middle conv
                            if self.is_deConv == 0:
                                self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(stream_out_chs[subs-1], stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                                stream_strides[subs], stream_paddings[subs]))
                            else:
                                self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(stream_out_chs[subs-1], stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                                stream_strides[subs], stream_paddings[subs]))
                            if stream_layers[subs]=="conv1d":
                                if BN_1D ==1:
                                    self.spat_temp_conv.add_module('BN_subs'+str(subs),nn.BatchNorm3d(stream_out_chs[subs]))
                            else:
                                if BN_2D ==1:
                                    self.spat_temp_conv.add_module('BN_subs'+str(subs),nn.BatchNorm3d(stream_out_chs[subs]))
                            
                            #self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))
                else: # BN_1D_last ==1
                    for subs in range(len(stream.kernels)):
                        if subs == 0 or subs ==len(stream.kernels)-1 : # first or last conv
                            if subs == 0: # first conv
                                if self.is_deConv == 0:
                                    self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(ch_in, stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                                    stream_strides[subs], stream_paddings[subs]))
                                else:
                                    self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(ch_in, stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                                    stream_strides[subs], stream_paddings[subs]))                                
                                #self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))
                            else: # last conv1d
                                if self.is_deConv == 0:
                                    self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(stream_out_chs[subs-1], ch_out, stream_kernel_sizes[subs], 
                                                                    stream_strides[subs], stream_paddings[subs]))
                                else:
                                    self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(stream_out_chs[subs-1], ch_out, stream_kernel_sizes[subs], 
                                                                    stream_strides[subs], stream_paddings[subs]))
                                if stream_layers[subs]=="conv1d":
                                    self.spat_temp_conv.add_module('BN_subs'+str(subs),nn.BatchNorm3d(ch_out))
                                else:
                                    assert (), 'The last layer must be Conv1D'
                                self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))
                        
                        else: # middle conv
                            if self.is_deConv == 0:
                                self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(stream_out_chs[subs-1], stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                                stream_strides[subs], stream_paddings[subs])) 
                            else:
                                self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(stream_out_chs[subs-1], stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                                stream_strides[subs], stream_paddings[subs]))                    
                            #self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else: # model_bn == 0
                for subs in range(len(stream.kernels)):
                    if subs == 0 or subs ==len(stream.kernels)-1 : # first or last conv
                        if subs == 0: # fisrt conv
                            if self.is_deConv == 0:
                                self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(ch_in, stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                                stream_strides[subs], stream_paddings[subs]))
                            else:
                                self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(ch_in, stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                                stream_strides[subs], stream_paddings[subs]))
                            #self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))
                        else: # last conv
                            if self.is_deConv == 0:
                                self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(stream_out_chs[subs-1], ch_out, stream_kernel_sizes[subs], 
                                                                stream_strides[subs], stream_paddings[subs]))
                            else:
                                self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(stream_out_chs[subs-1], ch_out, stream_kernel_sizes[subs], 
                                                            stream_strides[subs], stream_paddings[subs]))                                    
                            self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))
                    
                    else: # middle conv
                        if self.is_deConv == 0:
                            self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(stream_out_chs[subs-1], stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                            stream_strides[subs], stream_paddings[subs]))
                        else:
                                self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(stream_out_chs[subs-1], stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                            stream_strides[subs], stream_paddings[subs]))
                        #self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:   # lastBlock_WithBN ==0 No BN with last block
            for subs in range(len(stream.kernels)):
                if subs == 0 or subs ==len(stream.kernels)-1 : # first or last conv
                    if subs == 0: # fisrt conv2d
                        if self.is_deConv == 0:
                            self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(ch_in, stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                            stream_strides[subs], stream_paddings[subs]))
                        else:
                            self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(ch_in, stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                        stream_strides[subs], stream_paddings[subs]))
                        #self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))

                    else: # last conv
                        if self.is_deConv == 0:
                            self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(stream_out_chs[subs-1], ch_out, stream_kernel_sizes[subs], 
                                                            stream_strides[subs], stream_paddings[subs]))
                        else:
                            self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(stream_out_chs[subs-1], ch_out, stream_kernel_sizes[subs], 
                                                        stream_strides[subs], stream_paddings[subs]))
                        self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))
                
                else: # middle conv
                    if self.is_deConv == 0:
                        self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.Conv3d(stream_out_chs[subs-1], stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                        stream_strides[subs], stream_paddings[subs]))
                    else:
                        self.spat_temp_conv.add_module(stream_layers[subs]+'_subs'+str(subs), nn.ConvTranspose3d(stream_out_chs[subs-1], stream_out_chs[subs], stream_kernel_sizes[subs], 
                                                    stream_strides[subs], stream_paddings[subs]))
                    #self.spat_temp_conv.add_module('LeakyReLU_subs'+str(subs), nn.LeakyReLU(negative_slope=0.2, inplace=True))                                 

    def forward(self, x):
        x = self.spat_temp_conv(x)
        return x

if __name__ == '__main__':
    from factorizer import Factorizer
    import torch
    ch_in = 32
    ch_out = 1
    subspace_scale = 1
    stream_axes = [1]
    factorizer = Factorizer([[3,1,1],[1,3,1],[3,1,3],[3,3,1],[1,1,3]], ch_in, ch_out,subspace_scale, stream_axes)
    for i, stream in enumerate(factorizer.streams):
        model = SpatioTemporalConv(stream,ch_in,ch_out,model_bn=0,islastBlock=False, BN_1D=0, BN_2D=1)
    dummy_input = torch.randn(1, 32,24,46, 32,dtype=torch.float)
    print("org shape",dummy_input.shape)
    out = model(dummy_input)
    print("output", out.shape)
    #print(out)
    print(model)
    
    #for name, param in model.named_parameters():
    #    print(name,": ",param.requires_grad)
