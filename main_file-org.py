from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim 
import torch.utils.data
import numpy as np
import time
from models.StereoNet_Multi import StereoNet
from models.StereoNet_Multi import StereoNet as StereoNet_org
from models.StereoNet_Multi_SepConv import StereoNet as StereoNetSepConv
from models.StereoNet_Multi_FactorizedConv3D import StereoNet as StereoNetFactorized
from save_dataset_numpy import load_saved_numpy_dataset

import copy
from qtorch import FixedPoint, FloatingPoint
############
import cProfile 
########
from utils.folder_logs_init import *
from utils.save_load import load_dataset, save_best_lossess, save_chckpoint, load_checkpoint, load_quantized_model, save_losses, save_GL
from utils.test import test
from utils.train import *
from utils.stop_early import *
####
def select_model_mode(args,log,root_path):
    if args.mode == "train":
        model = StereoNet(k=3, r=args.stages-1, maxdisp=args.maxdisp)
        if args.model == 'cf_sepconv':
            log.info("-- model using stereonet with Separable Covolution for training--")
            model = StereoNetSepConv(k=3, r=args.stages-1, maxdisp=args.maxdisp) ### with separable conv
        elif args.model == 'cf_fact3d':
            log.info("-- model using stereonet with Factorized Conv3D for Cost Filtering Layer for training--")
            model = StereoNetFactorized(r=args.stages-1, use_skip=args.use_skip, initial_ch=args.initial_ch, num_convs_in_layers=args.num_convs_in_layers, initial_scale_factor=args.initial_scale_factor, 
                disp_offset=args.disp_offset,sub_pixel_acc=args.sub_pixel_acc, patch_index = args.patch_index, is_costvolume_4D=args.is_costvolume_4D, BN_2D_last=args.BN_2D_last, chout_costfiltring=args.chout_costfiltring, is_filter1_differ=args.is_filter1_differ,filter1_kernels=args.filter1_kernels,fact_kernels=args.fact_kernels,BN_1D = args.BN_1D, BN_2D = args.BN_2D,BN_1D_last=args.BN_1D_last, model_bn=args.model_bn, maxdisp=args.maxdisp) ### with factorized conv3d
        else: # org model
            log.info("-- model using Original StereoNet for training--")
        
        if args.cuda:
            model = nn.DataParallel(model)
            model.cuda()
        
        if args.dataset == "kitti": 
            #optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        if args.dataset == "sceneflow": ## to check it
            #model.apply(weights_init)
            #print('init with normal')
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
            #scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
    if args.mode == 'finetune':
        pretrainedModel_path = root_path + str(args.loadmodel)
        if os.path.isfile(pretrainedModel_path):
            model_org = StereoNet_org(k=3, r=args.stages-1, maxdisp=args.maxdisp)
            if args.cuda:
                model_org = nn.DataParallel(model_org)
                model_org.cuda()
            optimizer = optim.Adam(model_org.parameters(), lr=args.lr, betas=(0.9, 0.999))
            # optimizer = RMSprop(model.parameters(), lr=1e-3, weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            if args.model == 'cf_sepconv':
                log.info("-- model using stereonet with Separable Covolution for finetuning--")
                model = StereoNetSepConv(k=3, r=args.stages-1, maxdisp=args.maxdisp) ### with separable conv
                if args.cuda:
                    model = nn.DataParallel(model)
                    model.cuda()
                #model = mergingDifferentModels(args,model,pretrainedModel_path, log)
                log.info("-- pretrained model loaded for separable convolution for cost filtering layer --")
                #model =freezingtheModel(args,model,log)
            elif args.model == 'cf_fact3d':
                log.info("-- model using stereonet with factorized conv3d in the cost vlume filtering for finetuning--")
                #model = StereoNetFactorized(r=args.stages-1, initial_ch, num_convs_in_layers, initial_scale_factor, disp_offset,sub_pixel_acc, is_filter1_differ=args.is_filter1_differ,filter1_kernels=args.filter1_kernels,fact_kernels=args.fact_kernels,BN_1D = args.BN_1D, BN_2D = args.BN_2D,BN_1D_last=args.BN_1D_last, model_bn=args.model_bn, maxdisp=args.maxdisp) ### with factorized conv3d
                model = StereoNetFactorized(r=args.stages-1, use_skip=args.use_skip, initial_ch=args.initial_ch, num_convs_in_layers=args.num_convs_in_layers, initial_scale_factor=args.initial_scale_factor, 
                    disp_offset=args.disp_offset,sub_pixel_acc=args.sub_pixel_acc, patch_index = args.patch_index, is_costvolume_4D=args.is_costvolume_4D, BN_2D_last=args.BN_2D_last, chout_costfiltring=args.chout_costfiltring, is_filter1_differ=args.is_filter1_differ,filter1_kernels=args.filter1_kernels,fact_kernels=args.fact_kernels,BN_1D = args.BN_1D, BN_2D = args.BN_2D,BN_1D_last=args.BN_1D_last, model_bn=args.model_bn, maxdisp=args.maxdisp) ### with factorized conv3d
                if args.cuda:
                    model = nn.DataParallel(model)
                    model.cuda()
                #model = mergingDifferentModels(args,model,pretrainedModel_path, log)
                #model =freezingtheModel(args,model,log)
            else: # args.model == org
                log.info("-- model using the original stereonet --")
                model = StereoNet(k=3, r=args.stages-1, maxdisp=args.maxdisp)
                if args.cuda:
                    model = nn.DataParallel(model)
                    model.cuda()
                state_dict = torch.load(pretrainedModel_path)
                model.load_state_dict(state_dict['state_dict'])
                log.info("-- pretrained model was loaded for the original StereoNet --")
               
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  
            print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        else: # pretrained model not exist
            log.info("No ptratrained model exist in the path: "+pretrainedModel_path )
            exit()
    if args.mode == 'test':
        model = StereoNet(k=3, r=args.stages-1, maxdisp=args.maxdisp)
        if args.model == 'cf_sepconv':
            log.info("-- model using stereonet with Separable Covolution for testing--")
            model = StereoNetSepConv(k=3, r=args.stages-1, maxdisp=args.maxdisp) ### with separable conv
        elif args.model == 'cf_fact3d':
            log.info("-- model using stereonet with Factorized Conv3D for Cost Filtering Layer for testing--")
            #model = StereoNetFactorized(k=3, r=args.stages-1,is_filter1_differ=args.is_filter1_differ,filter1_kernels=args.filter1_kernels,fact_kernels=args.fact_kernels,BN_1D = args.BN_1D, BN_2D = args.BN_2D,BN_1D_last=args.BN_1D_last, model_bn=args.model_bn, maxdisp=args.maxdisp) ### with factorized conv3d
            model = StereoNetFactorized(r=args.stages-1, use_skip=args.use_skip, initial_ch=args.initial_ch, num_convs_in_layers=args.num_convs_in_layers, initial_scale_factor=args.initial_scale_factor, 
                disp_offset=args.disp_offset,sub_pixel_acc=args.sub_pixel_acc, patch_index = args.patch_index, is_costvolume_4D=args.is_costvolume_4D, BN_2D_last=args.BN_2D_last, is_filter1_differ=args.is_filter1_differ, chout_costfiltring=args.chout_costfiltring, filter1_kernels=args.filter1_kernels,fact_kernels=args.fact_kernels,BN_1D = args.BN_1D, BN_2D = args.BN_2D,BN_1D_last=args.BN_1D_last, model_bn=args.model_bn, maxdisp=args.maxdisp) ### with factorized conv3d
        else: # org model
            log.info("-- model using Original StereoNet for testing--")
        if args.cuda:
            model = nn.DataParallel(model)
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        # optimizer = RMSprop(model.parameters(), lr=1e-3, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) 
    return model,optimizer,scheduler

def main(args):
    args.stages = len(args.num_convs_in_layers)
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    #torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    for i in range(len(args.fact_kernels)):
        args.fact_kernels[i]=list(map(int, args.fact_kernels[i]))
    for i in range(len(args.filter1_kernels)):
        args.filter1_kernels[i]=list(map(int, args.filter1_kernels[i]))
    
    #args.num_convs_in_layers=list(map(int, args.num_convs_in_layers))
    #args.loss_weights=list(map(float, args.loss_weights))

    stages=args.stages ################2
    epoch_start = 1
    max_epo = 1
    no_improvement_epochs = 0   # for early stopping
    n_stop_epochs= 25#500       # for early stopping
    if args.dataset == "sceneflow":
        n_stop_epochs= 5
    best_error = +np.inf
    
    prev_return_avg_losses = []
    return_avg_losses =[]
    prev_sum_stages_losses=float('inf')
    losses_All_Stages={}
    losses_sum_All_Stages={}
    epes_All_Stages={}
    epes_sum_All_Stages={}
    test_train_losses={}
    test_train_EPEs={}
    test_train_sum_stages_losses={}
    #test_train_outliers_sumary={}
    test_train_outliers_sumary1={}
    test_train_outliers_sumary2={}
    test_train_outliers_sumary3={}
    ealryStoppedPaths=[]
    earlyStoppedEpochs=[]
    ### for validaion error early stopping
    split_epochs = 20
    if args.dataset == "sceneflow":
        split_epochs = 5
    start_avergaing = 1
    sumErrVal = 0
    sumErrTr =0
    end_averaging= split_epochs
    avgValErr_for_split_epochs=[]
    minValErr_for_split_epochs=[]
    avgTrErr_for_split_epochs=[]
    minTrErr_for_split_epochs=[]
    GL_Tr_for_split_epochs=[]   # generaliztion training loss for split epochs
    GL_Val_for_split_epochs = [] # generaliztion testing loss for split epochs
    temp_minValErr=float('inf')
    temp_minTrErr=float('inf')
    
    minLoss = {'loss0':float('inf'),'loss1':float('inf'),'loss2':float('inf'),'loss3':float('inf'),'sum_losses_stages':float('inf'),'test_sum_losses_stages':float('inf'),'checkpointPath':'none', 'epoch':1}
    
    timestr1, path_file_losses, path_file_GL,log,root_path,checkpoints_path,logs_path,test_results_path,valid_file_results = inialize_log_file(args)

    if args.load_numpy ==1:
        #filename = new_save_filepath + "/FlyingThings3D"
        TrainImgLoader = load_saved_numpy_dataset(args.datapath+"/train/",batch_size = args.train_bsize,shuffle = True,num_workers=12)
        TestImgLoader = load_saved_numpy_dataset(args.datapath+"/val/",batch_size= args.test_bsize,shuffle = False,num_workers=4)
    else:
        TrainImgLoader, TestImgLoader = load_dataset(args)
    
    model,optimizer,scheduler = select_model_mode(args,log,root_path)
    max_checkpoints_to_save = args.max_checkpoints_to_save
    threshold_overfit_epochs = args.threshold_overfit_epochs
    best_checkpoints = save_best_checkpoints(checkpoints_path,max_checkpoints_to_save)
    check_train_overfit = check_overfit(threshold_overfit_epochs)
    threshold_steadystate_epochs=threshold_overfit_epochs
    check_train_steadystate = check_steadyState(threshold_steadystate_epochs)
    
    ##############
    log.info("GPU is: " + torch.cuda.get_device_name(torch.cuda.current_device()))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    log.info('Number of images to train is: {} with train batch size: {}'.format(len(TrainImgLoader),args.train_bsize))
    log.info('Number of images to test is: {} with test batch size: {}'.format(len(TestImgLoader),args.test_bsize))
    ##############
    if args.mode == "finetune" or args.mode == "train":
        log.info("model loaded with k= 3 " +"and stages= "+ str(args.stages) )
        for x in range(stages):
            prev_return_avg_losses.append(float('inf'))
            return_avg_losses=[float('inf')]
        ## for resuming
        if args.resume == 1:
            resumeFile = root_path+ "/" + args.save_path+ args.resumeFile
            if os.path.isfile(resumeFile):
                model, return_avg_losses, ret_avg_train_epe_stage,optimizer, scheduler, epoch_start, current_lr,minLoss,saved_args,other_returned_values = load_checkpoint(
                    resumeFile,model,optimizer,scheduler)
                if len(other_returned_values) != 0:
                    best_checkpoints = other_returned_values[0]
                log.info("-- checkpoint loaded and the training is resumed from epoch "+ str(epoch_start)+" --")
            else:
                log.info("The resume file is: "+ resumeFile)
                log.info("-----The resume file is not exist------")
                exit()
        
        if args.with_quant ==1:
            model, float_model_dict = load_quantized_model(args,model)
        log.info(model)
        if args.resume == 1:
            log.info("*********The_resumed_learning_rate: {:.12f} ".format(optimizer.param_groups[0]['lr']))
        else:
            log.info("*********The_initial_learning_rate: {:.12f} ".format(optimizer.param_groups[0]['lr']))
        start_full_time = time.time()
        if args.epochs <=  epoch_start:
            if args.resume == 1:
                log.info("-----The model will resume from epoch "+str(epoch_start)+ ", so epochs value "+ str(args.epochs)+ " in argss file must be larger than "+ str(epoch_start) +" ------")
                exit()
            else:
                log.info("-----the model will start training or finetuning from epoch " + str(epoch_start)+", so epochs value "+ str(args.epochs)+ " in argss file must be larger than "+ str(epoch_start) +" ------")
                exit()
        for epoch in range(epoch_start, args.epochs+1):
            is_steadystate, steadystate_started_epoch = check_train_steadystate.get_steadystate_status()
            is_overfit, overfit_started_epoch = check_train_overfit.get_overfit_status()
            if is_overfit == 1:
                log.info("!!!!!!!!!!! We reached overfit for " + str(threshold_overfit_epochs) + " epochs. Overfit started at epoch: " + str(overfit_started_epoch))
                break
            elif is_steadystate ==1:
                log.info("!!!!!!!!!!! We reached steady state for " + str(threshold_steadystate_epochs) + " epochs. steady state started at epoch: " + str(steadystate_started_epoch))
                break
            else: # no over fit or steady state
                timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                log.info('############## This is %d-th epoch' % epoch +" ###############")
                epoch_train_start_time= time.time()
                
                if args.with_quant ==1:
                    return_avg_losses,return_sum_stages_losses,float_model_dict,return_avg_epes,return_sum_stages_epes = train(args,TrainImgLoader, model, optimizer, log, float_model_dict,epoch)
                else:
                    return_avg_losses,return_sum_stages_losses,float_model_dict,return_avg_epes,return_sum_stages_epes = train(args,TrainImgLoader, model, optimizer, log, None,epoch)
                
                epoch_train_end_time= time.time()
                losses_All_Stages[epoch]=return_avg_losses
                losses_sum_All_Stages[epoch]=return_sum_stages_losses
                epes_All_Stages[epoch]=return_avg_epes
                epes_sum_All_Stages[epoch]=return_sum_stages_epes
                
                if args.with_quant ==1:
                    checkName = args.model+'_fin_withQuant_'+args.dataset+'-'+args.datatype+'-'+timestr+'-epoch-'+str(epoch)+'-loss'+str(args.stages-1)+'-'+str(round(return_avg_losses[args.stages-1],3))+'-lossesSum-'+str(round(return_sum_stages_losses,3))+'.pth'
                else:
                    checkName = args.model+'_fin_'+args.dataset+'-'+args.datatype+'-'+timestr+'-epoch-'+str(epoch)+'-loss'+str(args.stages-1)+'-'+str(round(return_avg_losses[args.stages-1],3))+'-lossesSum-'+str(round(return_sum_stages_losses,3))+'.pth'
                
                if args.dataset == "kitti":
                    scheduler,optimizer = adjust_learning_rate(args,scheduler,optimizer, epoch,log)
                else:
                    scheduler,optimizer = adjust_learning_rate(args,scheduler,optimizer, epoch,log)
                    #scheduler.step()
                    #log.info("*********The_learning_rate is: {:.12f} ".format(optimizer.param_groups[0]['lr']))

                ##### validation for each epoch
                log.info('##### validation for epoch '+ str(epoch)+ '#####')
                epoch_test_start_time= time.time()
                #if epoch % 5 == 0:
                #test_return_avg_losses,test_return_avg_EPEs,mask_names,return_test_sum_stages_losses,return_outliers_sumary =test_from_training(args,TestImgLoader, model,log)
                test_return_avg_losses,test_return_avg_EPEs,mask_names,return_test_sum_stages_losses,return_outliers_sumary1,return_outliers_sumary2,return_outliers_sumary3 =test_from_training(args,TestImgLoader, model,log)
                epoch_test_end_time= time.time()

                test_train_losses[ epoch]=test_return_avg_losses
                test_train_EPEs[ epoch]=test_return_avg_EPEs
                test_train_sum_stages_losses[ epoch]=return_test_sum_stages_losses
                #test_train_outliers_sumary[epoch]= return_outliers_sumary
                test_train_outliers_sumary1[epoch]= return_outliers_sumary1
                test_train_outliers_sumary2[epoch]= return_outliers_sumary2
                test_train_outliers_sumary3[epoch]= return_outliers_sumary3
                ### saving losses
                save_losses(mask_names,optimizer,path_file_losses, epoch,stages,losses_All_Stages,epes_All_Stages,test_train_losses,test_train_EPEs,losses_sum_All_Stages,epes_sum_All_Stages,
                    test_train_sum_stages_losses,test_train_outliers_sumary1,test_train_outliers_sumary2,test_train_outliers_sumary3,epoch_train_end_time,epoch_train_start_time,epoch_test_end_time,epoch_test_start_time,scheduler)  
                
                #################
                savefilename = checkpoints_path + '/' + checkName
                minLoss = save_best_lossess(args,return_avg_losses,return_sum_stages_losses,return_avg_epes,return_sum_stages_epes,return_test_sum_stages_losses[0],minLoss,epoch,savefilename,model,optimizer,scheduler,best_checkpoints,check_train_overfit) 

                if epoch % args.checkpoint_save_thr == 0: # 100
                    print(savefilename)
                    save_chckpoint(args,model,return_avg_losses,return_avg_epes,epoch,optimizer,scheduler,minLoss,savefilename,best_checkpoints)

                # check if reached overfit or steady state
                reach_overfit_or_steadystate(args, prev_sum_stages_losses, return_sum_stages_losses,epoch,check_train_overfit,check_train_steadystate)

                prev_return_avg_losses = return_avg_losses[:]
                prev_sum_stages_losses = return_sum_stages_losses
                ############### calculate the generalaization training and testing error
                sumErrTr +=losses_sum_All_Stages[epoch]
                sumErrVal += test_train_sum_stages_losses[ epoch][0]
                
                if losses_sum_All_Stages[epoch] < temp_minTrErr:
                    temp_minTrErr = losses_sum_All_Stages[epoch]
                
                if test_train_sum_stages_losses[ epoch][0] < temp_minValErr:
                    temp_minValErr = test_train_sum_stages_losses[ epoch][0]
                
                if epoch % split_epochs == 0:
                    ####for training losss
                    avgTrErr_for_split_epochs,minTrErr_for_split_epochs,GL_Tr_for_split_epochs,temp_minTrErr,sumErrTr =get_avgErr_SPlitEpochs(
                        avgTrErr_for_split_epochs,minTrErr_for_split_epochs,GL_Tr_for_split_epochs,sumErrTr,split_epochs,temp_minTrErr)
                    ####for validation or testing loss
                    avgValErr_for_split_epochs,minValErr_for_split_epochs,GL_Val_for_split_epochs,temp_minValErr,sumErrVal =get_avgErr_SPlitEpochs(
                        avgValErr_for_split_epochs,minValErr_for_split_epochs,GL_Val_for_split_epochs,sumErrVal,split_epochs,temp_minValErr)
                    
                    start_avergaing = end_averaging + 1
                    end_averaging *= 2 
                    
                    save_GL(path_file_GL,avgTrErr_for_split_epochs[-1],minTrErr_for_split_epochs[-1],avgValErr_for_split_epochs[-1],
                        minValErr_for_split_epochs[-1],GL_Tr_for_split_epochs[-1],GL_Val_for_split_epochs[-1])
                    ############early stopping condition
                    best_model_dict={}
                    best_model_dict['temp_return_avg_losses']=return_avg_losses
                    best_model_dict['temp_optimizer_state_dict']=optimizer
                    best_model_dict['sheduler']=scheduler
                    best_model_dict['temp_min_loss']=minLoss
                    best_model_dict['temp_epoch']=epoch
                    best_model_dict['temp_savefilenameEarlyStopping']='none'
                    best_model = copy.deepcopy(model)
                    
                    best_model,no_improvement_epochs,best_model_dict,ealryStoppedPaths,best_error,earlyStoppedEpochs = stop_early(
                        args,model,optimizer,scheduler,log, avgValErr_for_split_epochs[-1],best_error,timestr,epoch,return_avg_losses, 
                        return_sum_stages_losses,minLoss, earlyStoppedEpochs,n_stop_epochs,no_improvement_epochs,best_model,ealryStoppedPaths,
                        best_model_dict,checkpoints_path,best_checkpoints,return_avg_epes)
            
        if args.with_quant ==1:
            model_save_path = checkpoints_path + '/' + 'checkpoint_withQuantize_finetune_kitti'+args.datatype+'-'+timestr+'-epoch-'+str(epoch)+'-loss'+str(args.stages-1)+'-'+str(round(return_avg_losses[args.stages-1],3))+'-lossesSum-'+str(round(return_sum_stages_losses,3))+'.pth'
            save_chckpoint(args,model,return_avg_losses,return_avg_epes,epoch,optimizer,scheduler,minLoss,model_save_path,best_checkpoints)
        
        if best_checkpoints.get_best_checkpoint() != None:
            log.info('#################The best checkpoint is: minLoss, best_path, best_sum_stages_losses ##########')
            min_loss_log=""
            
            for k,v in best_checkpoints.get_best_checkpoint()[0].items():
                if k == 'checkpointPath' or k == 'epoch':
                    min_loss_log = min_loss_log + ', '.join([k+": "+ str(v)+", "])
                else:
                    min_loss_log = min_loss_log + ', '.join([k+": "+ str(round(v,4))+", "])
            log.info('Min losses are: '+ min_loss_log)
            log.info("The name of the best checkpoint is: "+ best_checkpoints.get_best_checkpoint()[1])
            log.info("The minimum sum stages loss is: " + str(best_checkpoints.get_best_checkpoint()[2]))
        else:
            log.info('#################The best checkpoint is empty ##########')
        log.info('full_training_time: {: 3f} Hours'.format((time.time() - start_full_time) / 3600))

    else : ## Test ##
        infor_str = ''.join(['############# Start testing'])
        log.info(infor_str)
        testFile = root_path+ "/" + args.save_path+ args.testFile
        if os.path.isfile(testFile):
            ### loading saved values in the checkpoint
            if args.with_quant ==1:
                model, t_avg_train_loss_stages, avg_train_epe_stage,optimizer, scheduler, t_epoch, current_lr,temp_best_losses_stages,saved_args,other_returned_values = load_checkpoint(
                    testFile,model,optimizer,scheduler)
                if len(other_returned_values) != 0:
                    best_checkpoints = other_returned_values[0]

                if saved_args.with_quant ==0: # if the saved model not quantized previously
                    forward_num = FixedPoint(wl=args.quantWL, fl=args.quantFL, clamp=True, symmetric=False)
                    backward_num = FloatingPoint(exp=4, man=4)
                    #layerTypes=['conv','linear','pool','pad','activation','normalization','dropout','loss']
                    layerTypes=['activation','loss']
                    model = sequential_lower(model, layer_types=layerTypes, forward_number=forward_num)#, backward_number=backward_num)
                    quant_model_dict=quantize_model(model.state_dict(),args.quantWL,args.quantWL -args.quantFL)
                    model.load_state_dict(quant_model_dict)
            else:
                model, t_avg_train_loss_stages, t_avg_train_epe_stage, optimizer, scheduler, t_epoch, current_lr,temp_best_losses_stages,saved_args,other_returned_values = load_checkpoint(
                        testFile,model,optimizer,scheduler)
                if len(other_returned_values) != 0:
                    best_checkpoints = other_returned_values[0]
            if t_avg_train_epe_stage != None:
                log.info('Testing model at epoch {} with: sum_losses_stages is {:.3f} and sum_EPEs_stages is {:.3f}, current_learning_rate {:.10f} '.format(
                    t_epoch,sum( t_avg_train_loss_stages[x] for x in range(len(t_avg_train_loss_stages))), sum( t_avg_train_epe_stage[x] for x in range(len(t_avg_train_epe_stage))),current_lr))
            else:
                log.info('Testing model at epoch {} with: sum_losses_stages is {:.3f} and current_learning_rate {:.10f} '.format(
                    t_epoch,sum( t_avg_train_loss_stages[x] for x in range(len(t_avg_train_loss_stages))), current_lr))
            info_str3 = ', '.join(['Stage{} {:.3f}'.format(x, temp_best_losses_stages['loss'+str(x)]) for x in range(stages)])
            log.info('The best loss is for epoch' +  str(temp_best_losses_stages['epoch']) +' with losses: '+ info_str3)
            log.info('The sum of validation losses of this best checkpoint is: {:.5f}'.format(temp_best_losses_stages['test_sum_losses_stages'])) 
            log.info("*************** Saved Args in Checkpoint **************")
            log.info("The saved args in checkpoint are:")
            for key, value in sorted(vars(saved_args).items()):
                log.info(str(key) + ':' + str(value))
            log.info("*******************************************************")
            start_test_full_time = time.time()
            
            #total_outliers_summary,mask_names = test(args,TestImgLoader, model,test_results_path,log,valid_file_results)
            outliers_summary1,outliers_summary2,outliers_summary3,mask_names = test(args,TestImgLoader, model,test_results_path,log,valid_file_results)
        
            testing_time = time.time() - start_test_full_time
            log.info('The testing time is %d sec (%.3f min)' %(testing_time,testing_time/60))
            log.info('The result path is: '+ test_results_path)
            #log.info("Best epoch is {}, max accuracy perecnt is: {}".format(epoch, max_acc))
        else:
            log.info("There is no checkpoint to test")
            exit()

if __name__ == '__main__':

    main()
