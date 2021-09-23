from termcolor import colored
import numpy
import torch
from texttable import Texttable
####################################################################################################################################
# https://github.com/Xilinx/pytorch-quantization/blob/master/quantization/function/quantization_scheme.py
####################################################################################################################################
# This function assumes that the sign bit is included in L_I and the actual integer bitlength is L_I - 1
def s_quant_numpy(Tensor, bits=8, L_I=None,verbose=False):
        Tensor = numpy.array(Tensor)
        #Tensor[Tensor==0] = 1e-10
        if L_I is None:
                print(type(max(abs(Tensor.flatten()))))
                L_I = numpy.ceil(numpy.log2(1+numpy.floor(max(abs(Tensor.flatten())))))
                L_I = int(max(0,L_I))
                L_I = L_I + 1
        if L_I > bits:
                L_I = bits
        L_I = L_I * numpy.ones(Tensor.shape)
        L_F = bits - L_I
        prescale = 2.0**(L_F)                                            #default: 64
        postscale = 2.0**(-(L_F))                                        #default: 1/64
        signed_min = - 2.0**(L_I-1)                  #default: -2
        signed_max = - signed_min - postscale      #default: 2 - 1/64 => output range = [-2, 2-1/64]
        QTensor = Tensor * prescale
        QTensor = QTensor.round()
        QTensor = QTensor * postscale
        QTensor = numpy.clip(QTensor,signed_min,signed_max)
        if verbose:
                t = Texttable()
                t.set_precision(10)
                t_rows = [[]]
                for i in range(Tensor.flatten().shape[0]):
                        t_rows.append([i,Tensor.flatten()[i], QTensor.flatten()[i], bits, L_I.flatten()[i], L_F.flatten()[i]])
                t.add_rows(t_rows)
                t.header(['NO.','Value','Quantized','L_N','L_I','L_F'])
                print(t.draw())
        return QTensor
####################################################################################################################################
def u_quant_numpy(Tensor, bits=8, L_I=None, verbose=False):
        Tensor = numpy.array(Tensor)
        #Tensor[Tensor==0] = 1e-10
        if L_I is None:
                L_I = numpy.ceil(numpy.log2(1+numpy.floor(max(abs(Tensor.flatten())))))
                L_I = int(max(0,L_I))
        L_I = L_I * numpy.ones(Tensor.shape)
        L_F = bits - L_I 
        prescale = 2.0**(L_F)                                            #default: 64
        postscale = 2.0**(-(L_F))                                        #default: 1/64
        signed_min = 0                  #default: -2
        signed_max = 2.0**(L_I) - postscale      #default: 2 - 1/64 => output range = [-2, 2-1/64]
        QTensor = Tensor * prescale
        QTensor = QTensor.round()
        QTensor = QTensor * postscale
        QTensor = numpy.clip(QTensor,signed_min,signed_max)
        if verbose:
                t = Texttable()
                t.set_precision(10)
                t_rows = [[]]
                for i in range(Tensor.flatten().shape[0]):
                        t_rows.append([i,Tensor.flatten()[i], QTensor.flatten()[i], bits, L_I.flatten()[i], L_F.flatten()[i]])
                t.add_rows(t_rows)
                t.header(['NO.','Value','Quantized','L_N','L_I','L_F'])
                print(t.draw())
        return QTensor
####################################################################################################################################
# This function assumes that the sign bit is included in L_I and the actual integer bitlength is L_I - 1
def s_quant_PyTorch(Tensor, bits=8, L_I=None, verbose=False):
        #with torch.no_grad():
        #Tensor[Tensor==0] = 1e-10
        if L_I is None:
                TensorMax = torch.max(torch.abs(Tensor))
                TensorMax = TensorMax.float()
                L_I = torch.ceil(torch.log2(1+torch.floor(TensorMax)))
                L_I = int(max(0,L_I))
                L_I = L_I + 1
        if L_I > bits:
                L_I = bits
        ones_tensor = torch.ones(Tensor.shape)
        if Tensor.is_cuda :
                ones_tensor = ones_tensor.cuda()
        L_I = L_I * ones_tensor
        L_F = bits - L_I #- 1
        prescale = 2.0**(L_F)
        postscale = 2.0**(-(L_F))
        signed_min = - 2.0**(L_I-1)
        signed_max = - signed_min - postscale
        QTensor = Tensor * prescale
        QTensor = QTensor.round()
        QTensor = QTensor * postscale
        QTensor = torch.where(QTensor<signed_min,signed_min,QTensor)
        QTensor = torch.where(QTensor>signed_max,signed_max,QTensor)
        if verbose:
                t = Texttable()
                t.set_precision(10)
                t_rows = [[]]
                for i in range(Tensor.flatten().shape[0]):
                        t_rows.append([i,Tensor.flatten()[i], QTensor.flatten()[i], bits, L_I.flatten()[i], L_F.flatten()[i]])
                t.add_rows(t_rows)
                t.header(['NO.','Value','Quantized','L_N','L_I','L_F'])
                print(t.draw())
        return QTensor
####################################################################################################################################
def u_quant_PyTorch(Tensor, bits=8, L_I=None, verbose=False):
        #with torch.no_grad():
        #Tensor[Tensor==0] = 1e-10
        if L_I is None:
                TensorMax = torch.max(abs(Tensor))
                TensorMax = TensorMax.float()
                L_I = torch.ceil(torch.log2(1+torch.floor(TensorMax)))
                L_I = int(max(0,L_I))
        if L_I > bits:
                L_I = bits
        ones_tensor = torch.ones(Tensor.shape)
        if Tensor.is_cuda :
                ones_tensor = ones_tensor.cuda()
        L_I = L_I * ones_tensor
        L_F = bits - L_I
        prescale = 2.0**(L_F)
        postscale = 2.0**(-(L_F))
        signed_min = 0*ones_tensor
        signed_max = 2.0**(L_I) - postscale
        QTensor = Tensor * prescale
        QTensor = QTensor.round()
        QTensor = QTensor * postscale
        QTensor = torch.where(QTensor<signed_min,signed_min,QTensor)
        QTensor = torch.where(QTensor>signed_max,signed_max,QTensor)
        if verbose:
                t = Texttable()
                t.set_precision(10)
                t_rows = [[]]
                for i in range(Tensor.flatten().shape[0]):
                        t_rows.append([i,Tensor.flatten()[i], QTensor.flatten()[i], bits, L_I.flatten()[i], L_F.flatten()[i]])
                t.add_rows(t_rows)
                t.header(['NO.','Value','Quantized','L_N','L_I','L_F'])
                print(t.draw())
        return QTensor
####################################################################################################################################
def quantize_model_old(model_stat_dict,wl,il):
    ###########feature_extraction
    #for key in list(model_stat_dict.keys()):
    with torch.no_grad():
        for key, value in model_stat_dict.items():
            if "feature_extraction" in key:
                model_stat_dict[key].copy_( s_quant_PyTorch(value,wl, il))
                #print(key)
                #print(model_stat_dict[key])
    '''
    ########### Filter
    for key in list(model_stat_dict.keys()):
        if "filter" in key:
            print(key)
    ########### edge_aware_refinements.0 
    for key in list(model_stat_dict.keys()):
        if "edge_aware_refinements" in key:
            print(key)   
    exit()
    
    with torch.no_grad():
        #sd = model_stat_dict
        for key, value in model_stat_dict.items():
            #print(key)
            #temp = s_quant_PyTorch(value,wl, il) 
            #model_stat_dict[key]=temp
            model_stat_dict[key].copy_( s_quant_PyTorch(value,wl, il))
            #print(model_stat_dict[key])
        #model_stat_dict = sd  
    '''
    return model_stat_dict

def quantize_model(model_stat_dict,wl,il):
    with torch.no_grad():
        #sd = model_stat_dict
        for key, value in model_stat_dict.items():
            #print(key)
            #temp = s_quant_PyTorch(value,wl, il) 
            #model_stat_dict[key]=temp
            model_stat_dict[key].copy_( s_quant_PyTorch(value,wl, il))
            #print(model_stat_dict[key])
            #exit()
        #model_stat_dict = sd   
    return model_stat_dict
