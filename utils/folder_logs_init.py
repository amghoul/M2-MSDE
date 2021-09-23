from __future__ import print_function
import os
import time
import utils.logger as logger

def create_result_folders_org(args,modelName,stages ):
    if args.with_quant == 1:
        if args.dataset == "kitti":
            foldersToCreate=[ args.save_path, args.dataset+'_'+args.datatype+'_model_'+modelName+'_withQuant_'+str(stages)+'stages','checkpoints','finetune_train_logs','test_results']
        else:
            foldersToCreate=[ args.save_path, args.dataset+'_model_'+modelName+'_withQuant_'+str(stages)+'stages','checkpoints','finetune_train_logs','test_results']
    else:
        if args.dataset == "kitti":
            foldersToCreate=[ args.save_path, args.dataset+'_'+args.datatype+'_model_'+modelName+'_'+str(stages)+'stages','checkpoints','finetune_train_logs','test_results']
        else:
            foldersToCreate=[ args.save_path, args.dataset+'_model_'+modelName+'_'+str(stages)+'stages','checkpoints','finetune_train_logs','test_results']
    if (args.mode == "train" or args.mode == "finetune") and args.resume == 0:
        main_model_folder = foldersToCreate[0]+"/"+foldersToCreate[1]
        if os.path.isdir(main_model_folder):
            timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
            new_name = foldersToCreate[0]+"/"+foldersToCreate[1]+'-'+timestr
            os.rename(main_model_folder, new_name) 
        CHECK_FOLDER = os.path.isdir(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[2])
        if not CHECK_FOLDER:
            os.makedirs(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[2])
        CHECK_FOLDER = os.path.isdir(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[3])
        if not CHECK_FOLDER:
            os.makedirs(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[3])
        CHECK_FOLDER = os.path.isdir(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[4])
        if not CHECK_FOLDER:
            os.makedirs(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[4])
    else:
        CHECK_FOLDER = os.path.isdir(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[2])
        if not CHECK_FOLDER:
            os.makedirs(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[2])
        CHECK_FOLDER = os.path.isdir(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[3])
        if not CHECK_FOLDER:
            os.makedirs(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[3])
        CHECK_FOLDER = os.path.isdir(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[4])
        if not CHECK_FOLDER:
            os.makedirs(foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[4])
    
    current_directory = os.path.abspath(os.getcwd())
    root_path = current_directory+'/'+foldersToCreate[0]
    checkpoints_path = root_path +'/'+foldersToCreate[1]+"/"+foldersToCreate[2]
    logs_path = root_path +'/'+ foldersToCreate[1]+"/"+foldersToCreate[3]
    test_results_path = root_path +'/'+foldersToCreate[1]+"/"+foldersToCreate[4]
    return current_directory,checkpoints_path,logs_path,test_results_path

def create_result_folders(args,modelName,stages ):
    current_directory = os.path.abspath(os.getcwd()) #+"/drive/MyDrive/StereoNet-Last-DFKI" ## for colab
    if args.with_quant == 1:
        if args.dataset == "kitti":
            foldersToCreate=[ args.save_path, args.dataset+'_'+args.datatype+'_model_'+modelName+'_withQuant_'+str(stages)+'stages','checkpoints','finetune_train_logs','test_results']
        else:
            foldersToCreate=[ args.save_path, args.dataset+'_model_'+modelName+'_withQuant_'+str(stages)+'stages','checkpoints','finetune_train_logs','test_results']
    else:
        if args.dataset == "kitti":
            foldersToCreate=[ args.save_path, args.dataset+'_'+args.datatype+'_model_'+modelName+'_'+str(stages)+'stages','checkpoints','finetune_train_logs','test_results']
        else:
            foldersToCreate=[ args.save_path, args.dataset+'_model_'+modelName+'_'+str(stages)+'stages','checkpoints','finetune_train_logs','test_results']
    if (args.mode == "train" or args.mode == "finetune") and args.resume == 0:
        main_model_folder = current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]
        if os.path.isdir(main_model_folder):
            timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
            new_name = current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+'-'+timestr
            os.rename(main_model_folder, new_name) 
        CHECK_FOLDER = os.path.isdir(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[2])
        if not CHECK_FOLDER:
            os.makedirs(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[2])
        CHECK_FOLDER = os.path.isdir(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[3])
        if not CHECK_FOLDER:
            os.makedirs(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[3])
        CHECK_FOLDER = os.path.isdir(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[4])
        if not CHECK_FOLDER:
            os.makedirs(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[4])
    else:
        CHECK_FOLDER = os.path.isdir(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[2])
        if not CHECK_FOLDER:
            os.makedirs(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[2])
        CHECK_FOLDER = os.path.isdir(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[3])
        if not CHECK_FOLDER:
            os.makedirs(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[3])
        CHECK_FOLDER = os.path.isdir(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[4])
        if not CHECK_FOLDER:
            os.makedirs(current_directory+ "/"+foldersToCreate[0]+"/"+foldersToCreate[1]+"/"+foldersToCreate[4])
    
    root_path = current_directory+'/'+foldersToCreate[0]
    checkpoints_path = root_path +'/'+foldersToCreate[1]+"/"+foldersToCreate[2]
    logs_path = root_path +'/'+ foldersToCreate[1]+"/"+foldersToCreate[3]
    test_results_path = root_path +'/'+foldersToCreate[1]+"/"+foldersToCreate[4]
    return current_directory,checkpoints_path,logs_path,test_results_path

def inialize_log_file(args):
    stages=args.stages
    modelName = args.model
    timestr1 = time.strftime("%Y_%m_%d-%H_%M_%S")
    root_path,checkpoints_path,logs_path,test_results_path=create_result_folders(args,modelName,stages)
    
    if args.with_quant ==1:
        path_file_GL= logs_path +'/' + args.model+'-'+args.mode +'-'+args.dataset+'-'+args.datatype+'-withQuantize-GL-'+str(args.stages)+'stages-'+timestr1+'.txt'
        path_file_losses = logs_path +'/' + args.model+'-'+args.mode +'-'+args.dataset+'-'+args.datatype+'-withQuantize-losses-'+str(args.stages)+'stages-'+timestr1+'.txt'
        valid_file_results = test_results_path +'/' + args.model+'-'+args.mode +'-'+args.dataset+'-'+args.datatype+'-withQuantize-losses-'+str(args.abs_thr)+'-px-'+str(args.stages)+'stages-'+timestr1+'.txt'
    else:
        path_file_GL= logs_path +'/' + args.model+'-'+args.mode +'-'+args.dataset+'-'+args.datatype+'-GL-'+str(args.stages)+'stages-'+timestr1+'.txt'
        path_file_losses = logs_path +'/' + args.model+'-'+args.mode +'-'+args.dataset+'-'+args.datatype+'-losses-'+str(args.stages)+'stages-'+timestr1+'.txt'
        valid_file_results = test_results_path +'/' + args.model+'-'+args.mode +'-'+args.dataset+'-'+args.datatype+'-losses-'+str(args.abs_thr)+'-px-'+str(args.stages)+'stages-'+timestr1+'.txt'
    if args.dataset == "kitti":
        if args.datatype == '2015':
            mask_names=['allw','allwo','bgw','bgwo','fgw','fgwo','occw','occwo','noccw','noccwo'] # w: with rel_threshold, wo: without rel_threshold
        else: #args.datatype == '2012':
            mask_names=['allw','allwo','occw','occwo','noccw','noccwo'] # w: with rel_threshold, wo: without rel_threshold
    else:
        mask_names=['allw','allwo','occw','occwo','noccw','noccwo'] # w: with rel_threshold, wo: without rel_threshold
    thr_list = [1,2,3]
    if args.stages ==1 and args.mode != 'test':
        with open(path_file_losses, 'a+') as f:
            f.write("epoch:Tr_loss0:Tr_epe0:Te_loss0:Te_EPE0:")
            for x in range(args.stages):
                for i in range(len(mask_names)):
                    for t in range(len(thr_list)):
                        f.write(mask_names[i])
                        f.write("_thr_%d" % (thr_list[t]))
                        f.write("_st_%d:" % (x))
            f.write("Tr_sum_losses:Tr_sum_epes:Te_sum_losses:Tr_time(s):Te_time(s):LR\n")
            f.close()
    elif args.stages ==2 and args.mode != 'test':
        with open(path_file_losses, 'a+') as f:
            f.write("epoch:Tr_loss0:Tr_epe0:Tr_loss1:Tr_epe1:Te_loss0:Te_loss1:Te_EPE0:Te_EPE1:")
            for x in range(args.stages):
                for i in range(len(mask_names)):
                    for t in range(len(thr_list)):
                        f.write(mask_names[i])
                        f.write("_thr_%d" % (thr_list[t]))
                        f.write("_st_%d:" % (x))
            f.write("Tr_sum_losses:Tr_sum_epes:Te_sum_losses:Tr_time(s):Te_time(s):LR\n")
            f.close()
    elif args.stages ==3 and args.mode != 'test':
        with open(path_file_losses, 'a+') as f:
            f.write("epoch:Tr_loss0:Tr_epe0:Tr_loss1:Tr_epe1:Tr_loss2:Tr_epe2:Te_loss0:Te_loss1:Te_loss2:Te_EPE0:Te_EPE1:Te_EPE2:")
            for x in range(args.stages):
                for i in range(len(mask_names)):
                    for t in range(len(thr_list)):
                        f.write(mask_names[i])
                        f.write("_thr_%d" % (thr_list[t]))
                        f.write("_st_%d:" % (x))
            f.write("Tr_sum_losses:Tr_sum_epes:Te_sum_losses:Tr_time(s):Te_time(s):LR\n")
            f.close()
    elif args.stages ==4 and args.mode != 'test':
        with open(path_file_losses, 'a+') as f:
            f.write("epoch:Tr_loss0:Tr_epe0:Tr_loss1:Tr_epe1:Tr_loss2:Tr_epe2:Tr_loss3:Tr_epe3:Te_loss0:Te_loss1:Te_loss2:Te_loss3:Te_EPE0:Te_EPE1:Te_EPE2:Te_EPE3:")
            for x in range(args.stages):
                for i in range(len(mask_names)):
                    for t in range(len(thr_list)):
                        f.write(mask_names[i])
                        f.write("_thr_%d" % (thr_list[t]))
                        f.write("_st_%d:" % (x))
            f.write("Tr_sum_losses:Tr_sum_epes:Te_sum_losses:Tr_time(s):Te_time(s):LR\n")
            f.close()
    if args.mode != 'test':
        with open(path_file_GL, 'a+') as f_GL:
                f_GL.write("avgTrErr:minTrErr:avgValErr:minValErr:GL_tr:GL_val\n")
                f_GL.close()
    if args.mode == "test":
        if args.with_quant ==1:
            log = logger.setup_logger(test_results_path +'/' + args.model+'-'+args.mode +'-'+args.dataset+'-'+args.datatype+'-withQuantize'+'-'+str(args.abs_thr)+'-px-'+str(args.stages)+'stages-'+timestr1+'.log')
        else:
            log = logger.setup_logger(test_results_path +'/' + args.model+'-'+args.mode +'-'+args.dataset+'-'+args.datatype+'-'+str(args.abs_thr)+'-px-'+str(args.stages)+'stages-'+timestr1+'.log')
    else:
        if args.resume == 0 and args.with_quant ==1:    
            log = logger.setup_logger(logs_path +'/' + args.model+'-'+args.mode +'-'+args.dataset+'-'+args.datatype+'-withQuantize'+'-'+str(args.stages)+'stages-'+timestr1+'.log')
        elif args.resume == 1 and args.with_quant ==0:
            log = logger.setup_logger(logs_path +'/' + args.model+'-'+args.mode +'-resume-'+args.dataset+'-'+args.datatype+'-'+str(args.stages)+'stages-'+args.model+'-'+timestr1+'.log')
        elif args.resume == 1 and args.with_quant ==1:    
            log = logger.setup_logger(logs_path +'/' + args.model+'-'+args.mode +'-resume-'+args.dataset+'-'+args.datatype+'-withQuantize'+'-'+str(args.stages)+'stages-'+timestr1+'.log')
        else:
            log = logger.setup_logger(logs_path +'/' + args.model+'-'+args.mode +'-'+args.dataset+'-'+args.datatype+'-'+str(args.stages)+'stages-'+args.model+'-'+timestr1+'.log')
    log.info("========= The " + args.mode + "ing was started =========")
    if(args.model == 'org'):
        log.info('========= The original StereoNet was loaded =========' )  
    if(args.model == 'cf_sepconv'):
        log.info('========= The StereoNet with Separable Covolution in Cost Filtering Layer was loaded (cf_sepconv)=========' ) 
    if args.model == 'cf_fact3d':
        if args.model_bn == 1:
            if args.is_filter1_differ == 1:
                if args.BN_1D_last == 0:
                    log.info('========= Model ' + args.model + ' was loaded (filter 1 is differ) with number of subspaces is '+ str(len(args.fact_kernels)+len(args.filter1_kernels)-1) + ' + last Conv1D. BN_1D is ' + str(args.BN_1D)+ ' and BN_2D is ' + str(args.BN_2D))  
                else:
                    log.info('========= Model ' + args.model + ' was loaded (filter 1 is differ) with number of subspaces is '+ str(len(args.fact_kernels)+len(args.filter1_kernels)-1) + ' + last Conv1D. BN enabled for the last conv1D only')  
                log.info("The used kernels are: "+ "filter1 kernels: "+ str(args.filter1_kernels)+ ", Kernels for the remaining filters are: "+ str(args.fact_kernels))
            else:
                if args.BN_1D_last == 0:
                    log.info('========= Model ' + args.model + ' was loaded with number of subspaces is '+ str(len(args.fact_kernels)-1) + ' + last Conv1D.  BN_1D is ' + str(args.BN_1D)+ ' and BN_2D is ' + str(args.BN_2D))
                else:
                    log.info('========= Model ' + args.model + ' was loaded with number of subspaces is '+ str(len(args.fact_kernels)-1) + ' + last Conv1D. BN enabled for the last conv1D only')
                log.info("The used kernels are: "+ str(args.fact_kernels))
        else:
            if args.is_filter1_differ == 1:
                log.info('========= Model ' + args.model + ' was loaded (filter 1 is differ) with number of subspaces is '+ str(len(args.fact_kernels)+len(args.filter1_kernels)-1) + ' + last Conv1D, with no Batch Normalized used for whole the model')  
                log.info("The used kernels are: "+ "filter1 kernels: "+ str(args.filter1_kernels)+ ", Kernels for the remaining filters are: "+ str(args.fact_kernels))
            else:
                log.info('========= Model ' + args.model + ' was loaded with number of subspaces is '+ str(len(args.fact_kernels)-1) + ' + last Conv1D, with no Batch Normalized used for whole the model')
                log.info("The used kernels are: "+ str(args.fact_kernels))
        
        log.info("novel Stereonet arguments are: ")
        log.info("initial_ch: "+ str(args.initial_ch) + ", " +  "initial_scale_factor: "+ str(args.initial_scale_factor) + ", " + "disp_offset: "+ str(args.disp_offset) )
        log.info("sub_pixel_acc: "+ str(args.sub_pixel_acc) + ", " +  "patch_index: "+ str(args.patch_index) + ", " + "chout_costfiltring: "+ str(args.chout_costfiltring) )
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ':' + str(value))
    return timestr1, path_file_losses, path_file_GL,log,root_path,checkpoints_path,logs_path,test_results_path,valid_file_results
