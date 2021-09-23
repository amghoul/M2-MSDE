from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
import copy
from .utils import *
########## for qunatization
from utils.FinalQuant import *

def adjust_learning_rate(args,scheduler,optimizer, epoch,log):
    if args.with_quant ==0:
        if args.dataset == "kitti":
            if epoch <= 200:
                lr = 0.001
            else: # epoch > 200
                if (scheduler.get_last_lr()[0]> 0.00001):
                    if epoch % 50 == 0:
                        scheduler.step() # will adjust learning rate
                        lr = scheduler.get_last_lr()[0]
                    else:
                        lr = scheduler.get_last_lr()[0]
                else:
                    lr = scheduler.get_last_lr()[0]
        else: #scenflow
            if epoch <= 5:
                lr = 0.001
            else: # epoch > 200
                if (scheduler.get_last_lr()[0]> 0.00001):
                    if epoch % 1 == 0:
                        scheduler.step() # will adjust learning rate
                        if (scheduler.get_last_lr()[0]< 0.00001):
                            lr = 0.00001
                        else:
                            lr = scheduler.get_last_lr()[0]
                        #lr = scheduler.get_last_lr()[0]
                    else:
                        lr = scheduler.get_last_lr()[0]
                else:
                    lr = scheduler.get_last_lr()[0]
    else: #args.with_quant ==1:
        if epoch <= 2000:
            lr = 0.01
        elif epoch > 2000 and epoch <= 3000:
            lr = 0.001
        elif epoch > 3000 and epoch <= 3500:
            lr = .0005
        else:
            lr =  0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    log.info("*********The_learning_rate is: {:.12f} ".format(param_group['lr']))
    return scheduler,optimizer

def train(args,dataloader, model, optimizer, log, float_model_dict=None,epoch=1):
    initial_scale_factor = args.initial_scale_factor
    stages = args.stages #default 4, used 2
    with_quant =  args.with_quant
    losses = [AverageMeter() for _ in range(stages)]
    sum_stages_losses= []
    length_loader = len(dataloader)
    counter = 0

    model.train()
    avg_sum_loss_all_stages_epoch=0
    for batch_idx, (imgL, imgR, disp_L, disp_L_occ_noc, obj_map_img,left_path,right_path) in enumerate(dataloader):
        ##disp_L_occ_noc : if dataset kitti then this variable fron noc, if dataset is FT3D then this variable is for occ
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()

        if with_quant == 1:
            imgL = s_quant_PyTorch(imgL,args.quantWL,args.quantWL -args.quantFL )
            imgR = s_quant_PyTorch(imgR,args.quantWL,args.quantWL -args.quantFL )

        disp_L = disp_L.float().cuda()
        
        outputs = model(imgL, imgR)
        outputs = [torch.squeeze(output, 1) for output in outputs]
        disp_L_scales=[]
        disp_L = disp_L.unsqueeze(1)
        disp_L_h,disp_L_w = disp_L.size(-2),disp_L.size(-1)
        
        #print("disp_l",disp_L.size() )
        for s in range(stages):
            out_h, out_w = outputs[s].size(-2), outputs[s].size(-1)
            disp_L_scales.append(F.interpolate(disp_L,size=(out_h, out_w),mode='bilinear', align_corners=True).squeeze(1) * out_w/disp_L_w )
        '''
        for s in range(stages):
            print("siz########e",outputs[s].size(), disp_L_scales[s].size())
            print(torch.max(disp_L), torch.max(disp_L_scales[s]))
        exit()
        '''
        ###############
        '''
        if args.dataset == "kitti":
            loss = [smooth_L1_loss(outputs[0],disp_L, args)]
            for i in range(len(outputs)-1):
                loss.append(smooth_L1_loss(outputs[i+1],disp_L, args))
        else:
            loss = [GERF_loss(outputs[0],disp_L, args)]
            for i in range(len(outputs)-1):
                loss.append(GERF_loss(outputs[i+1],disp_L, args))
        '''
        scale = (initial_scale_factor * (2 ** (2 -0)))
        loss = [GERF_loss(outputs[0],disp_L_scales[0], args,scale)]
        for i in range(len(outputs)-1):
            scale = (initial_scale_factor * (2 ** (2 -i+1)))
            loss.append(GERF_loss(outputs[i+1],disp_L_scales[i+1], args, scale))
        ########   
        counter +=1
        loss_all = sum(loss)/(args.itersize)
        sum_stages_losses.append(loss_all.item())
        
        loss_all.backward()
        if with_quant == 1:
            model.load_state_dict(float_model_dict)
        
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0

        for idx in range(stages):
            losses[idx].update(loss[idx].item(),args.train_bsize)
            #losses[idx].update(loss[idx].item()/args.loss_weights[idx])
        if with_quant == 1: 
            float_model_dict = copy.deepcopy(model.state_dict())
            quant_model_dict = copy.deepcopy(model.state_dict())
            quant_model_dict=quantize_model(quant_model_dict,args.quantWL,args.quantWL -args.quantFL) # loop for quantizing weight parameters
            model.load_state_dict(quant_model_dict)
        
        avg_sum_loss_all_stages=0
        for x in range(stages):
            avg_sum_loss_all_stages +=losses[x].avg

        if batch_idx % args.print_freq == 0: #if batch_idx == 0 or batch_idx == 6:
            info_str = ['Stage{} {:.3f} ( {:.3f} )'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]
            info_str = ' '.join(info_str)
            info_sum_losses = ' sum_stages_losses {:.3f} ( {:.3f} )'.format(sum_stages_losses[-1], avg_sum_loss_all_stages)
            info_sum_losses = ''.join(info_sum_losses)
            log.info('Epoch {} [{}/{}] {} {}'.format(epoch, batch_idx, length_loader, info_str,info_sum_losses))
    avg_sum_loss_all_stages_epoch += avg_sum_loss_all_stages   
    info_str = ' '.join(['Stage{} {:.3f}'.format(x, losses[x].avg) for x in range(stages)])
    info_sum_losses = ' sum_stages_losses {:.3f} ( {:.3f} )'.format(sum(sum_stages_losses)/length_loader, avg_sum_loss_all_stages_epoch)
    log.info('Average_train_loss: ' + info_str +info_sum_losses)
    return_avg_losses=[]
    for x in range(stages):
        return_avg_losses.append(losses[x].avg)
    
    return return_avg_losses,sum(sum_stages_losses)/length_loader,float_model_dict

def test_from_training(args,dataloader, model, log):
    stages = args.stages
    
    REL_THRESH = args.rel_thr
    ABS_THRESH=args.abs_thr
    Outliers_rate=[]
    total_Outliers_rate=[]
    
    EPES_summary = {}
    losses_dict={}
    test_sum_stages_losses={}
    #outliers_summary = defaultdict(list)
    outliers_summary1 = defaultdict(list)
    outliers_summary2 = defaultdict(list)
    outliers_summary3 = defaultdict(list)
    
    if args.dataset == "kitti":
        if args.datatype == '2015':
            mask_names=['allw','allwo','bgw','bgwo','fgw','fgwo','occw','occwo','noccw','noccwo'] # w: with rel_threshold, wo: without rel_threshold
        else: #args.datatype == '2012':
            mask_names=['allw','allwo','occw','occwo','noccw','noccwo'] # w: with rel_threshold, wo: without rel_threshold
    else:
        mask_names=['allw','allwo','occw','occwo','noccw','noccwo'] # w: with rel_threshold, wo: without rel_threshold
    
    length_loader = len(dataloader)
    
    for x in range(stages):
        Outliers_rate.append(0)
    
    for item in mask_names:
        #outliers_summary[item]=[AverageMeter() for _ in range(stages)]#[0,0,0,0]
        ##########
        outliers_summary1[item]=[AverageMeter() for _ in range(stages)]#[0,0,0,0]
        outliers_summary2[item]=[AverageMeter() for _ in range(stages)]#[0,0,0,0]
        outliers_summary3[item]=[AverageMeter() for _ in range(stages)]#[0,0,0,0]
        ############
        EPES_summary[item]=[AverageMeter() for _ in range(stages)]
        losses_dict[item]=[AverageMeter() for _ in range(stages)]
        test_sum_stages_losses[item]=0.0
    model.eval()
    
    count = 0
    for batch_idx, (imgL, imgR, disp_L,disp_L_occ_noc,obj_map_crop_img,left_path,right_path) in enumerate(dataloader):
        count +=1
        #from matplotlib import pyplot as plt   
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()

        for i in range(len(mask_names)):
            for x in range(stages):
                Outliers_rate[x]=0
                loss=[]
            with torch.no_grad():
                outputs = model(imgL, imgR)
                outputs_size=[]
                for x in range(stages):
                    outputs_size.append(outputs[x].size())
                masks_list,disp_L_scales = get_masks_GT_scales(args,outputs_size,disp_L,disp_L_occ_noc,obj_map_crop_img)
                mask = masks_list[i]
                for x in range(stages):
                    if len(disp_L_scales[x][mask[x]]) == 0:
                        #EPES[x].update(0)
                        Outliers_rate[x]=0
                        continue
                    output = torch.squeeze(outputs[x], 1)
                    #print("output",output.size(), "mask", mask[x].size(),"disp_L_scales[x]",disp_L_scales[x].size() )
                    EPES_summary[mask_names[i]][x].update((output[mask[x]] - disp_L_scales[x][mask[x]]).abs().mean(),args.test_bsize)
                    '''
                    if mask_names[i][-1] == 'w': 
                        outliers_summary[mask_names[i]][x].update( Outliers(mask,output,disp_L,ABS_THRESH,REL_THRESH),args.test_bsize)
                    else:
                        outliers_summary[mask_names[i]][x].update( Outliers(mask,output,disp_L,ABS_THRESH,REL_THRESH, False),args.test_bsize)
                    '''
                    ###################### calculate outliers for threshold 1,2 and 3
                    if mask_names[i][-1] == 'w': 
                        outliers_summary1[mask_names[i]][x].update( Outliers(mask[x],output,disp_L_scales[x],1,REL_THRESH),args.test_bsize)
                        outliers_summary2[mask_names[i]][x].update( Outliers(mask[x],output,disp_L_scales[x],2,REL_THRESH),args.test_bsize)
                        outliers_summary3[mask_names[i]][x].update( Outliers(mask[x],output,disp_L_scales[x],3,REL_THRESH),args.test_bsize)

                    else:
                        outliers_summary1[mask_names[i]][x].update( Outliers(mask[x],output,disp_L_scales[x],1,REL_THRESH, False),args.test_bsize)
                        outliers_summary2[mask_names[i]][x].update( Outliers(mask[x],output,disp_L_scales[x],2,REL_THRESH, False),args.test_bsize)
                        outliers_summary3[mask_names[i]][x].update( Outliers(mask[x],output,disp_L_scales[x],3,REL_THRESH, False),args.test_bsize)

                    #######################
                    '''
                    if args.dataset == "kitti":
                        loss.append(smooth_L1_loss_mask(output,disp_L,mask,args))
                    else:
                        loss.append(GERF_loss_mask(output,disp_L, mask, args))
                    '''
                    loss.append(GERF_loss_mask(output,disp_L_scales[x], mask[x], args))
                    losses_dict[mask_names[i]][x].update(loss[x],args.test_bsize)
            test_sum_stages_losses[mask_names[i]]+=sum(loss)
    
    info_test = ', '.join(['#############Start_testing from training']) 
    info_outl="Thr1 --> "
    info_out2="Thr2 --> "
    info_out3="Thr3 --> "             
    for i in range(len(mask_names)):
        if mask_names[i][-1] == 'w':
            info_str_epe = ' '.join(['Stage{} ( avg_EPE {:.2f} - avg_loss {:.2f} )'.format(x, EPES_summary[mask_names[i]][x].avg, losses_dict[mask_names[i]][x].avg) for x in range(stages)])
            #info_str_outl = ' '.join(['Stage{} ({:.2f} )'.format(x, outliers_summary[mask_names[i]][x].avg*100) for x in range(stages)])
            ##################
            info_str_outl = ' '.join(['Stage{} ({:.2f} )'.format(x, outliers_summary1[mask_names[i]][x].avg*100) for x in range(stages)])
            info_str_out2 = ' '.join(['Stage{} ({:.2f} )'.format(x, outliers_summary2[mask_names[i]][x].avg*100) for x in range(stages)])
            info_str_out3 = ' '.join(['Stage{} ({:.2f} )'.format(x, outliers_summary3[mask_names[i]][x].avg*100) for x in range(stages)])
            #########
            info_sum_losses = ' sum_stages_losses {:.3f}'.format(test_sum_stages_losses[mask_names[i]]/length_loader)
            log.info('D1-{} Avg_test_EPE '.format(mask_names[i][:-1]) + info_str_epe+info_sum_losses)
            #log.info('D1-{} Avg_outliers '.format(mask_names[i][:-1]) + info_str_outl)
            ###########
            log.info('D1-{} Avg_outliers '.format(mask_names[i][:-1]) + info_outl + info_str_outl)
            log.info('D1-{} Avg_outliers '.format(mask_names[i][:-1]) + info_out2 + info_str_out2)
            log.info('D1-{} Avg_outliers '.format(mask_names[i][:-1]) + info_out3 + info_str_out3)
            ##########
            
    return_avg_losses=[]
    return_avg_EPEs=[]
    return_test_sum_stages_losses=[]
    #return_outliers_sumary=[]
    return_outliers_sumary1=[]
    return_outliers_sumary2=[]
    return_outliers_sumary3=[]
    for i in range(len(mask_names)):
        if mask_names[i][-1] == 'w':
            return_avg_losses.append([losses_dict[mask_names[i]][x].avg for x in range(stages)])
            return_avg_EPEs.append([EPES_summary[mask_names[i]][x].avg for x in range(stages)])
            #return_outliers_sumary.append([outliers_summary[mask_names[i]][x].avg.item()*100 for x in range(stages)])
            return_test_sum_stages_losses.append(test_sum_stages_losses[mask_names[i]]/length_loader)
        ##############
        return_outliers_sumary1.append([outliers_summary1[mask_names[i]][x].avg*100 for x in range(stages)])
        return_outliers_sumary2.append([outliers_summary2[mask_names[i]][x].avg*100 for x in range(stages)])
        return_outliers_sumary3.append([outliers_summary3[mask_names[i]][x].avg*100 for x in range(stages)])
        #############

    return return_avg_losses,return_avg_EPEs,mask_names,return_test_sum_stages_losses,return_outliers_sumary1,return_outliers_sumary2,return_outliers_sumary3 #return_outliers_sumary
    

