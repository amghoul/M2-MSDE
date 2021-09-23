## without using lateral layers
#https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
import math

class LayerScale(nn.Module): # contents of each layer scal: default 2 * (conv2d+BN+Relu)
    def __init__(self, in_planes, planes, strides):
        super(LayerScale, self).__init__()
        
        layers = OrderedDict()
        for i in range(len(strides)):
            if i ==0:
                layers["conv2d"+str(i)] = nn.Conv2d(in_planes, planes, kernel_size=3, stride=strides[i], padding=1, bias=False)
            else:
                layers["conv2d"+str(i)] = nn.Conv2d(planes, planes, kernel_size=3, stride=strides[i], padding=1,bias=False)
                
            layers["bn2d"+str(i)] = nn.BatchNorm2d(planes)
            layers["relu"+str(i)] = nn.ReLU()
    
        self.layer_convs = nn.Sequential(layers)
        
    def forward(self, x):
        out = self.layer_convs(x)
        return out

class FPN(nn.Module):
    def __init__(self, initial_ch, block, num_convs_in_layers,initial_scale_factor=4):
        super(FPN, self).__init__()
        #num_convs_in_layers: it is a list consists of number of conv2d layers in each layer scalee, and the length of this list is the number of scales
        #initial_ch: number of channels that to be input to the FPN after applying conv2d to the input 
        self.initial_scale_factor = initial_scale_factor # the first scale factor to begin our FPN network scaling
        self.in_planes = initial_ch # self.in_planes is the number of input channels for every layer
        
        # conv2d + BN on input image before to be input to the FPN
        layers = OrderedDict()
        no_input_reduce_layers = math.log2(initial_scale_factor)
        if initial_scale_factor > 2:
            assert no_input_reduce_layers % 2 == 0, "####### initial_scale_factor value should be power of 2"
            for i in range(int(no_input_reduce_layers)-1): # determine number of conv2d layers used to reduce the original input according to initial_scale_factor value
                if i==0:
                    layers["conv2d"+str(i)] = nn.Conv2d(3, initial_ch, kernel_size=7, stride=2, padding=3, bias=False)
                else:
                    layers["conv2d"+str(i)] = nn.Conv2d(initial_ch, initial_ch, kernel_size=7, stride=2, padding=3, bias=False)
                layers["bn2d"+str(i)] = nn.BatchNorm2d(initial_ch)
        else:
            assert initial_scale_factor > 1, "####### initial_scale_factor value should be larger than 1"
            layers["conv2d"+str(0)] = nn.Conv2d(3, initial_ch, kernel_size=7, stride=1, padding=3, bias=False)
            layers["bn2d"+str(0)] = nn.BatchNorm2d(initial_ch)
        layers["mpool2d"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers["relu"] = nn.ReLU()
        self.input_reduce_layer = nn.Sequential(layers)
    
        # Bottom-up layers
        # down_top_layers[0] contains the largest feature map scale, down_top_layers[2]: the smallest feature map scale
        self.down_top_layers=nn.ModuleList()
        no_layers = len(num_convs_in_layers)
        for lay_id in range(no_layers):            
            if lay_id == 0:
                self.down_top_layers.append(self._make_layer(block,  self.in_planes, initial_ch * (2 ** lay_id), num_convs_in_layers[lay_id],stride=1))
            else:
                self.down_top_layers.append(self._make_layer(block,  self.in_planes, initial_ch * (2 ** lay_id), num_convs_in_layers[lay_id],stride=2))
            
            self.in_planes = initial_ch * (2 ** lay_id)
        
        self.top_down_upsample_ayers=nn.ModuleList()
        for upsample_lay_id in range(no_layers-1,0,-1):
            upsample_in_planes = initial_ch * (2 ** (upsample_lay_id))
            upsample_out_planes = initial_ch * (2 ** (upsample_lay_id-1))
            self.top_down_upsample_ayers.append(nn.ConvTranspose2d(upsample_in_planes, upsample_out_planes, 3, stride = 2, padding=1))
        # top_down Smooth layers
        # top_down_smoothlayers[0] contains the mid feature map scale, top_down_smoothlayers[1]: the largest feature map scale
        self.top_down_smoothlayers=nn.ModuleList()
        for smooth_lay_id in range(no_layers-1,-1,-1):
            smooth_lay_planes = initial_ch * (2 ** (smooth_lay_id))
            self.top_down_smoothlayers.append(nn.Conv2d(smooth_lay_planes, smooth_lay_planes, kernel_size=1, stride=1, padding=0))
        
    def _make_layer(self, block, inplanes, planes, num_convs, stride):
        strides = [stride] + [1]*(num_convs-1)
        return block(inplanes, planes, strides)

    def _upsample_add(self, x, y,lay_ind):
        _,Cy,H,W = y.size()
        _,Cx,_,_ = x.size()
        return self.top_down_upsample_ayers[lay_ind](x,output_size=(H,W)) + y
        #nn.ConvTranspose2d(Cx, Cy, 3, stride = 2, padding=1)(x) 
        #return nn.functional.interpolate(self.top_down_upsample_ayers[lay_ind](x), size=(H,W), mode='nearest') + y 

    def forward(self, x):
        no_layers = len(self.down_top_layers)
        init_input_features=self.input_reduce_layer(x)
        # Bottom-up
        bot_up_features = {}  ## bot_up_features[0] --> contains the largest scale feature map
        scale_layer=[] # contains the scale used in every feature scale layer, forexample [4,8,16]
        for i in range(no_layers):
            if i == 0:
                bot_up_features[i] = self.down_top_layers[i](init_input_features)
            else:
                bot_up_features[i] = self.down_top_layers[i](bot_up_features[i-1])
            scale_layer.append(self.initial_scale_factor * (2 ** i))
        
        #top_down upsample + add + smoothing
        top_down_features={}
        for i in range(no_layers):
            if i==0:
                top_down_features[i] = self.top_down_smoothlayers[i](bot_up_features[no_layers-1-i])
            else:
                top_down_features[i] = self.top_down_smoothlayers[i](self._upsample_add(bot_up_features[no_layers - i],bot_up_features[no_layers -1 - i],i-1))
  
        return top_down_features,  scale_layer[::-1] # top_down_features[0] --> contains the smallest scale feature map

if __name__ == '__main__':
    def test():
        initial_ch=4
        num_convs_in_layers=[2,2,2]
        FPN_net = FPN(initial_ch,LayerScale, num_convs_in_layers,initial_scale_factor=4)
        input = torch.randn(1,3,600,900)
        fms, scale_layer = FPN_net(Variable(input))
        print("input size:", input.size())
        for k,fm in fms.items():
            print(fm.size(),scale_layer[k])
        print("scale_layer: ", scale_layer)
        '''
        print(FPN_net)
        from torchsummary import summary
        summary(FPN_net, (3,600,900))
        '''
    test()