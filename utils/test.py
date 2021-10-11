#test-v04
from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from .utils import *
from .disp_to_color import *
import time


def test(args,dataloader, model, test_results_path,log,valid_file_results):
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
    #allwo-thr-%d-EPE-%d-st-%d
    #allwo-thr-%d-outl-%d-st-%d
    mask_names_concat=""
    for i in mask_names:
        for x in range(stages):
            mask_names_concat+= i+"-EPE-st"+str(x)+":"
        for x in range(stages):
            for t in range(len(thr_list)):
                mask_names_concat+= i+"-thr-"+str(thr_list[t]) +"-Out-st"+str(x)+":"
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
            disp_L = disp_L.float().cuda() #
            obj_map_crop_img = obj_map_crop_img.float().cuda() #

            if args.dataset == "kitti":
                disp_L = disp_L.float().cuda() #
                disp_L_noc = disp_L_occ_noc.float().cuda() #
                mask_all = (disp_L < args.maxdisp) & (disp_L > 0)
                mask_occ= (abs(disp_L - disp_L_noc) > 0) *  mask_all
                mask_noc= (disp_L_noc > 0) * mask_all
                if args.datatype == '2015':
                    mask_bg = (obj_map_crop_img == 0) * mask_all
                    mask_fg = (obj_map_crop_img > 0) * mask_all
                    masks_list=[mask_all,mask_all,mask_bg,mask_bg,mask_fg,mask_fg, mask_occ,mask_occ,mask_noc,mask_noc]
                else:
                    masks_list=[mask_all,mask_all, mask_occ,mask_occ,mask_noc,mask_noc]
            else: # FlyingThings3D
                disp_L = disp_L.float().cuda() * -1 # multiply disp_l by -1 because its values are negative
                disp_L_occ = disp_L_occ_noc.float().cuda() #
                #mask_all = disp_L > 0
                mask_all = (disp_L < args.maxdisp) & (disp_L > 0)
                mask_occ = (disp_L_occ == 255)
                mask_noc = (disp_L_occ == 0)
                masks_list=[mask_all,mask_all, mask_occ,mask_occ,mask_noc,mask_noc]

            
            for i in range(len(mask_names)):
                mask = masks_list[i]
                concat_EPE_stages=""
                concat_outl_stages=""
                for x in range(stages):
                    Outliers_rate[x]=0
                with torch.no_grad():
                    '''
                    total_time = 0.0
                    for i in range(10):
                        outputs = model(imgL, imgR)
                    for i in range(300):
                        torch.cuda.synchronize()
                        starter_ft= time.perf_counter()
                        outputs = model(imgL, imgR)
                        torch.cuda.synchronize()
                        ender_ft= time.perf_counter()
                        total_time = total_time + (ender_ft-starter_ft)
                        print("time in ", str(i),": ", str(ender_ft-starter_ft))
                    print("average time: ", str(total_time/300))
                    exit()
                    '''
                    outputs = model(imgL, imgR)
                    for x in range(stages):
                        if len(disp_L[mask]) == 0:
                            Outliers_rate[x]=0
                            EPES_summary[mask_names[i]][x].update(0)
                            concat_EPE_stages += str(0)
                            continue
                        output = torch.squeeze(outputs[x], 1)
                        concat_EPE_stages += str(round(((output[mask] - disp_L[mask]).abs().mean()/args.test_bsize).item(),3))+":"
                        #f.write("batch_id:EPE/outlier:"+mask_names_concat+"left_path\n")
                        #f.write("{0:.3f}:".format(losses_All_Stages[epoch][x]))
                        EPES_summary[mask_names[i]][x].update((output[mask] - disp_L[mask]).abs().mean(),args.test_bsize)
                    
                    for x in range(stages):
                        if len(disp_L[mask]) == 0:
                            concat_outl_stages += str(0)+":"
                            outliers_summary1[mask_names[i]][x].update( 0,args.test_bsize)
                            concat_outl_stages += str(0)+":"
                            outliers_summary2[mask_names[i]][x].update( 0,args.test_bsize)
                            concat_outl_stages += str(0)+":"
                            outliers_summary3[mask_names[i]][x].update( 0,args.test_bsize)

                            continue
                        output = torch.squeeze(outputs[x], 1)
                        if mask_names[i][-1] == 'o':  # without
                            #concat_outl_stages += str(round(( temp_out *100/args.test_bsize).item(),3))+":"
                            #outliers_summary[mask_names[i]][x].update( Outliers(mask,output,disp_L,ABS_THRESH,REL_THRESH),args.test_bsize)
                            ####
                            temp_out1 = Outliers(mask,output,disp_L,1,REL_THRESH, False)
                            concat_outl_stages += str(round((temp_out1 * 100/args.test_bsize).item(),3))+":"
                            outliers_summary1[mask_names[i]][x].update( temp_out1,args.test_bsize)
                            
                            temp_out2 = Outliers(mask,output,disp_L,2,REL_THRESH, False)
                            concat_outl_stages += str(round((temp_out2 * 100/args.test_bsize).item(),3))+":"
                            outliers_summary2[mask_names[i]][x].update( temp_out2,args.test_bsize)

                            temp_out3 = Outliers(mask,output,disp_L,3,REL_THRESH, False)
                            concat_outl_stages += str(round((temp_out3 * 100/args.test_bsize).item(),3))+":"
                            outliers_summary3[mask_names[i]][x].update( temp_out3,args.test_bsize)
                            
                        else: # with
                            #concat_outl_stages += str(round((Outliers(mask,output,disp_L,ABS_THRESH,REL_THRESH, False)*100/args.test_bsize).item(),3))+":"
                            #outliers_summary[mask_names[i]][x].update( Outliers(mask,output,disp_L,ABS_THRESH,REL_THRESH, False),args.test_bsize)
                            ########
                            temp_out1 = Outliers(mask,output,disp_L,1,REL_THRESH)
                            concat_outl_stages += str(round((temp_out1 * 100/args.test_bsize).item(),3))+":"
                            outliers_summary1[mask_names[i]][x].update( temp_out1,args.test_bsize)

                            temp_out2 = Outliers(mask,output,disp_L,2,REL_THRESH)
                            concat_outl_stages += str(round((temp_out2 * 100/args.test_bsize).item(),3))+":"
                            outliers_summary2[mask_names[i]][x].update( temp_out2,args.test_bsize)

                            temp_out3 = Outliers(mask,output,disp_L,3,REL_THRESH)
                            concat_outl_stages += str(round((temp_out3 * 100/args.test_bsize).item(),3))+":"
                            outliers_summary3[mask_names[i]][x].update( temp_out3,args.test_bsize)
                            
                    f.write(concat_EPE_stages)
                    f.write(concat_outl_stages)
            f.write(left_path[0].split("/")[-1])
            f.write("\n")
            if args.dataset == "kitti":
                save_images(outputs,disp_L,left_path,right_path,batch_idx,ABS_THRESH,args,test_results_path)
            else:
                if batch_idx == 0 or batch_idx == 10 or batch_idx == 50 or batch_idx == 100 or batch_idx == 150 or \
                    batch_idx == 160 or batch_idx == 180 or batch_idx == 200 or batch_idx == 220 or batch_idx == 240 or \
                    batch_idx == 260 or batch_idx == 280 or batch_idx == 300 or batch_idx == 320 or batch_idx == 340:
                    save_images(outputs,disp_L,left_path,right_path,batch_idx,ABS_THRESH,args,test_results_path)
            '''
            #save images
            if args.dataset == "kitti":
                if ABS_THRESH ==3:
                    save_images(outputs,disp_L,left_path,right_path,batch_idx,ABS_THRESH,args,test_results_path)
            else:
                #if batch_idx % 500 == 0:
                #if batch_idx % 5 == 0:
                if ABS_THRESH ==3:
                    save_images(outputs,disp_L,left_path,right_path,batch_idx,ABS_THRESH,args,test_results_path)
            '''
        f.close()
    with open(valid_file_results, 'a+') as f:
        f.write("All:")
        for i in range(len(mask_names)):
            info_str_epe_file=""
            info_str3_file=""
            for x in range(stages):
                info_str_epe_file += str(round(EPES_summary[mask_names[i]][x].avg.item(),3)) + ":"
            for x in range(stages):
                #info_str3_file += str(round((outliers_summary[mask_names[i]][x].avg.item())*100,3))+":"
                info_str3_file += str((round(outliers_summary1[mask_names[i]][x].avg.item(),3))*100)+":"
                info_str3_file += str((round(outliers_summary2[mask_names[i]][x].avg.item(),3))*100)+":"
                info_str3_file += str((round(outliers_summary3[mask_names[i]][x].avg.item(),3))*100)+":"

            f.write(info_str_epe_file+info_str3_file)
        f.write("All\n")
        f.close()
    
        
    for i in range(len(mask_names)):
        if mask_names[i][-1] == 'w':
            info_str_epe = ' '.join(['Stage{} {:.3f}'.format(x, EPES_summary[mask_names[i]][x].avg) for x in range(stages)])
            log.info('D1-{} Avg_test_EPE '.format(mask_names[i][:-1]) + info_str_epe)
    
    for i in range(len(mask_names)):
        #info_str3 = ' '.join(['Stage{} {:.2f}'.format(x, (outliers_summary[mask_names[i]][x].avg)*100) for x in range(stages)])
        info_str_out_1 = ' '.join(['Stage{} {:.3f}'.format(x, ((outliers_summary1[mask_names[i]][x].avg)*100)) for x in range(stages)])
        info_str_out_2 = ' '.join(['Stage{} {:.3f}'.format(x, ((outliers_summary2[mask_names[i]][x].avg)*100)) for x in range(stages)])
        info_str_out_3 = ' '.join(['Stage{} {:.3f}'.format(x, ((outliers_summary3[mask_names[i]][x].avg)*100)) for x in range(stages)])
        if mask_names[i][-1] == 'w':
            thr = 1
            log.info('D1-{} Avg_Outliers_{}-px '.format(mask_names[i][:-1],thr) + info_str_out_1 )
            thr = 2
            log.info('D1-{} Avg_Outliers_{}-px '.format(mask_names[i][:-1],thr) + info_str_out_2 )
            thr = 3
            log.info('D1-{} Avg_Outliers_{}-px '.format(mask_names[i][:-1],thr) + info_str_out_3 )
        else:
            thr = 1
            log.info('D1-{} Avg_Outliers_{}-px_wihout_rel_thre '.format(mask_names[i][:-2],thr) + info_str_out_1 )
            thr = 2
            log.info('D1-{} Avg_Outliers_{}-px_wihout_rel_thre '.format(mask_names[i][:-2],thr) + info_str_out_2 )
            thr = 3
            log.info('D1-{} Avg_Outliers_{}-px_wihout_rel_thre '.format(mask_names[i][:-2],thr) + info_str_out_3 )
   
    #return outliers_summary,mask_names
    return outliers_summary1,outliers_summary2,outliers_summary3,mask_names
