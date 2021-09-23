from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from .utils import *
from .disp_to_color import *

def test(args,dataloader, model, test_results_path,log,valid_file_results):
    stages = args.stages

    REL_THRESH = args.rel_thr
    ABS_THRESH=args.abs_thr   
    Outliers_rate=[]
    EPES_summary = {}
    losses_dict={}
    test_sum_stages_losses={}
    #outliers_summary = defaultdict(list)
    outliers_summary1 = defaultdict(list)
    outliers_summary2 = defaultdict(list)
    outliers_summary3 = defaultdict(list)
    thr_list = [1,2,3]
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
    info_info = '\t'.join(['The total number of images to test is: {} for {} images per test batch size : {} '.format(length_loader*args.test_bsize, length_loader,args.test_bsize)])
    log.info(info_info)
    ############
    # i_thr_%d_st_%d
    #EPE-st-%d
    mask_names_concat=""
    for i in mask_names:
        for x in range(stages):
            mask_names_concat+= i+"-EPE-st"+str(x)+":"
        for x in range(stages):
            mask_names_concat+= i+"-Outl-st"+str(x)+":"
    #####
    for x in range(args.stages):
        for i in range(len(mask_names)):
            for t in range(len(thr_list)):
                f.write(mask_names[i])
                f.write("_thr_%d" % (thr_list[t]))
                f.write("_st_%d:" % (x))
    #####
    with open(valid_file_results, 'a+') as f:
        f.write("batch_id:"+mask_names_concat+"left_path\n")
        f.close()
    ######
    with open(valid_file_results, 'a+') as f:
        for batch_idx, (imgL, imgR, disp_L,disp_L_occ_noc,obj_map_crop_img,left_path,right_path) in enumerate(dataloader):
            print("Testing: "+ str(batch_idx) + "/"+ str(length_loader))
            f.write("%d:" % (batch_idx))
            count +=1
            imgL = imgL.float().cuda()
            imgR = imgR.float().cuda()
            
            for i in range(len(mask_names)):
                concat_EPE_stages=""
                concat_outl_stages=""
                for x in range(stages):
                    Outliers_rate[x]=0
                with torch.no_grad():
                    outputs = model(imgL, imgR)
                    outputs_size=[]
                    for x in range(stages):
                        outputs_size.append(outputs[x].size())
                    masks_list,disp_L_scales = get_masks_GT_scales(args,outputs_size,disp_L,disp_L_occ_noc,obj_map_crop_img)
                    mask = masks_list[i]
                    for x in range(stages):
                    
                        if len(disp_L_scales[x][mask[x]]) == 0:
                            EPES[x].update(0)
                            Outliers_rate[x]=0
                            continue
                        output = torch.squeeze(outputs[x], 1)
                        concat_EPE_stages += str(round(((output[mask] - disp_L[mask]).abs().mean()/args.test_bsize).item(),3))+":"
                        #f.write("batch_id:EPE/outlier:"+mask_names_concat+"left_path\n")
                        #f.write("{0:.3f}:".format(losses_All_Stages[epoch][x]))
                        EPES_summary[mask_names[i]][x].update((output[mask] - disp_L[mask]).abs().mean(),args.test_bsize)
                        if mask_names[i][-1] == 'w': 
                            concat_outl_stages += str(round((Outliers(mask,output,disp_L,ABS_THRESH,REL_THRESH)*100/args.test_bsize).item(),3))+":"
                            outliers_summary[mask_names[i]][x].update( Outliers(mask,output,disp_L,ABS_THRESH,REL_THRESH),args.test_bsize)
                        else:
                            concat_outl_stages += str(round((Outliers(mask,output,disp_L,ABS_THRESH,REL_THRESH, False)*100/args.test_bsize).item(),3))+":"
                            outliers_summary[mask_names[i]][x].update( Outliers(mask,output,disp_L,ABS_THRESH,REL_THRESH, False),args.test_bsize)
                    f.write(concat_EPE_stages)
                    f.write(concat_outl_stages)
            f.write(left_path[0].split("/")[-1])
            f.write("\n")
            #save images
            if args.dataset == "kitti":
                if ABS_THRESH ==3:
                    save_images(outputs,disp_L,left_path,right_path,batch_idx,ABS_THRESH,args,test_results_path)
            else:
                #if batch_idx % 500 == 0:
                #if batch_idx % 5 == 0:
                if ABS_THRESH ==3:
                    save_images(outputs,disp_L,left_path,right_path,batch_idx,ABS_THRESH,args,test_results_path)
        f.close()
    with open(valid_file_results, 'a+') as f:
        f.write("All:")
        for i in range(len(mask_names)):
            info_str_epe_file=""
            info_str3_file=""
            for x in range(stages):
                info_str_epe_file += str(round(EPES_summary[mask_names[i]][x].avg.item(),3)) + ":"
                info_str3_file += str(round((outliers_summary[mask_names[i]][x].avg.item())*100,3))+":"
            f.write(info_str_epe_file+info_str3_file)
        f.write("All\n")
        f.close()
    
        
    for i in range(len(mask_names)):
        if mask_names[i][-1] == 'w':
            info_str_epe = ' '.join(['Stage{} {:.2f}'.format(x, EPES_summary[mask_names[i]][x].avg) for x in range(stages)])
            log.info('D1-{} Avg_test_EPE '.format(mask_names[i][:-1]) + info_str_epe)
    
    for i in range(len(mask_names)):
        info_str3 = ' '.join(['Stage{} {:.2f}'.format(x, (outliers_summary[mask_names[i]][x].avg)*100) for x in range(stages)])
        if mask_names[i][-1] == 'w':
            log.info('D1-{} Avg_Outliers_{}-px '.format(mask_names[i][:-1],ABS_THRESH) + info_str3 )
        else:
            log.info('D1-{} Avg_Outliers_{}-px_wihout_rel_thre '.format(mask_names[i][:-2],ABS_THRESH) + info_str3 )
   
    return outliers_summary,mask_names#[mask_names[0]]#total_Outliers_rate
