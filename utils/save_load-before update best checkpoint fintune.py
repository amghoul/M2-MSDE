from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data
import copy
import os
from qtorch import FixedPoint, FloatingPoint
from qtorch.auto_low import sequential_lower, lower
from .FinalQuant import *

def save_best_lossess(args,return_avg_losses,return_sum_stages_losses,return_avg_epes,return_sum_stages_epes,minLoss,epoch,savefilename,model,optimizer,scheduler,best_checkpoints,check_train_overfit) :
    if return_sum_stages_losses < minLoss['sum_losses_stages']:
        minLoss['loss0'] = return_avg_losses[0]
        minLoss['epe0'] = return_avg_epes[0]
        if args.stages >=2:
            minLoss['loss1'] = return_avg_losses[1]
            minLoss['epe1'] = return_avg_epes[1]
        if args.stages >=3:
            minLoss['loss2'] = return_avg_losses[2]
            minLoss['epe2'] = return_avg_epes[2]
        if args.stages ==4:
            minLoss['loss3'] = return_avg_losses[3]
            minLoss['epe3'] = return_avg_epes[3]    
        minLoss['checkpointPath'] = savefilename
        minLoss['epoch'] = epoch
        minLoss['sum_losses_stages'] = return_sum_stages_losses
        minLoss['sum_epes_stages'] = return_sum_stages_epes
        best_checkpoints.update(args, model,return_avg_losses,return_avg_epes,epoch,optimizer,scheduler,minLoss,best_checkpoints)
        #check_train_overfit.reset()
    #if return_sum_stages_losses > minLoss['sum_losses_stages']:
    #    check_train_overfit.update(args, epoch)

    return minLoss

def reach_overfit_or_steadystate(args, prev_sum_stages_losses, return_sum_stages_losses,epoch,check_train_overfit,check_train_steadystate):
    round_value = 4
    if round(return_sum_stages_losses,round_value) > round(prev_sum_stages_losses,round_value):
        check_train_overfit.update(args, epoch)
    else:
        check_train_overfit.reset()
    if round(return_sum_stages_losses,round_value) == round(prev_sum_stages_losses,round_value):
        check_train_steadystate.update(args, epoch)
    else:
        check_train_steadystate.reset()

def save_chckpoint(args, model,return_avg_losses,return_avg_epes,epoch,optimizer,scheduler,minLoss,savefilename,best_checkpoints):
    torch.save({
        'state_dict': model.state_dict(),
        'avg_train_loss_stage': return_avg_losses,
        'avg_train_epe_stage': return_avg_epes,
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'sheduler' : scheduler.state_dict(),
        'current_learning_rate':optimizer.param_groups[0]['lr'],
        'best_losses_stages': minLoss,
        'saved_args': args,
        'best_checkpoints':best_checkpoints
        }, savefilename)

def load_checkpoint(checkpoint_file,model,optimizer,scheduler):
    checkpoint = torch.load(checkpoint_file)
    other_returned_values=[]
    model.load_state_dict(checkpoint['state_dict'])
    avg_train_loss_stage = checkpoint['avg_train_loss_stage']
    if 'avg_train_epe_stage' in checkpoint.keys():
        avg_train_epe_stage = checkpoint['avg_train_epe_stage']
    else:
        avg_train_epe_stage = None
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['sheduler'])
    epoch_start = checkpoint['epoch']+1
    current_lr=checkpoint['current_learning_rate']
    minLoss=checkpoint['best_losses_stages']
    saved_args=checkpoint['saved_args']
    if 'best_checkpoints' in checkpoint.keys():
        other_returned_values.append(checkpoint['best_checkpoints'])
    return model, avg_train_loss_stage, avg_train_epe_stage ,optimizer, scheduler, epoch_start, current_lr,minLoss,saved_args,other_returned_values
        

def load_quantized_model(args,model):
        forward_num = FixedPoint(wl=args.quantWL, fl=args.quantFL, clamp=True, symmetric=False)
        backward_num = FloatingPoint(exp=4, man=4)
        #layerTypes=['conv','linear','pool','pad','activation','normalization','dropout','loss']
        layerTypes=['activation','loss']
        model = sequential_lower(model, layer_types=layerTypes, forward_number=forward_num)#, backward_number=backward_num)
        float_model_dict = copy.deepcopy(model.state_dict())
        quant_model_dict = copy.deepcopy(model.state_dict())
        quant_model_dict=quantize_model(quant_model_dict,args.quantWL,args.quantWL -args.quantFL) # loop for quantizing weight parameters
        model.load_state_dict(quant_model_dict)
        return model, float_model_dict

def save_losses(mask_names,optimizer,path_file_losses, epoch,stages,losses_All_Stages,epes_All_Stages,test_train_losses,test_train_EPEs,losses_sum_All_Stages,epes_sum_All_Stages,
    test_train_sum_stages_losses,test_train_outliers_sumary1,test_train_outliers_sumary2,test_train_outliers_sumary3,epoch_train_end_time,epoch_train_start_time,epoch_test_end_time,epoch_test_start_time,scheduler):
    with open(path_file_losses, 'a') as f:
        f.write("%d:" % (epoch))
        for i in range(4):
            for x in range(stages):
                if i ==0:
                    f.write("{0:.3f}:".format(losses_All_Stages[epoch][x]))
                    f.write("{0:.3f}:".format(epes_All_Stages[epoch][x]))
                if i ==1:
                    f.write("{0:.3f}:".format(test_train_losses[epoch][0][x]))
                if i ==2:
                    f.write("{0:.3f}:".format(test_train_EPEs[epoch][0][x]))
                if i ==3:
                    for j in range(len(mask_names)):
                        #f.write("{0:.3f}:".format(test_train_outliers_sumary[epoch][0][x]))## return only the D1-allw
                        #mask_names[i]][x]
                        f.write("{0:.3f}:".format(test_train_outliers_sumary1[epoch][j][x]))## return only the D1-allw
                        f.write("{0:.3f}:".format(test_train_outliers_sumary2[epoch][j][x]))## return only the D1-allw
                        f.write("{0:.3f}:".format(test_train_outliers_sumary3[epoch][j][x]))## return only the D1-allw
        
        f.write("{0:.3f}:".format(losses_sum_All_Stages[epoch]))
        f.write("{0:.3f}:".format(epes_sum_All_Stages[epoch]))        
        f.write("{0:.3f}:".format(test_train_sum_stages_losses[epoch][0]))
        f.write("{0:.3f}:".format(epoch_train_end_time-epoch_train_start_time))
        f.write("{0:.3f}:".format(epoch_test_end_time-epoch_test_start_time))
        f.write("{0:.10f}".format(optimizer.param_groups[0]['lr']))
        f.write("\n")
        f.close()

def save_GL(path_file_GL,avgTrEr,minTrErr,avgValErr,minValErr,GL_Tr,GL_Val):
    with open(path_file_GL, 'a') as f:
        f.write("{0:.3f}:".format(avgTrEr))
        f.write("{0:.3f}:".format(minTrErr))
        f.write("{0:.3f}:".format(avgValErr))
        f.write("{0:.3f}:".format(minValErr))
        f.write("{0:.3f}:".format(GL_Tr))
        f.write("{0:.3f}".format(GL_Val))
        f.write("\n")
        f.close()

def load_dataset(args):
    if args.dataset == "kitti":
        if args.datatype == '2015':
            from dataloader import KITTIloader2015 as ls
            from dataloader import KITTILoader as DA
            train_left_img, train_right_img, train_left_disp,train_left_disp_noc, test_left_img, test_right_img, test_left_disp,test_left_disp_noc,train_mask_obj_map,test_mask_obj_map = ls.dataloader(
            args.datapath)
            
        else:# args.datatype == '2012':
            from dataloader import KITTIloader2012 as ls
            from dataloader import KITTILoader1 as DA
            train_left_img, train_right_img, train_left_disp,train_left_disp_noc, test_left_img, test_right_img, test_left_disp,test_left_disp_noc = ls.dataloader(
            args.datapath)
            
    else: ##sceneflow dataset
        from dataloader import listflowfile as lt  ## change import fie for scenflow dataset
        #from dataloader import SecenFlowLoader1 as DA
        from dataloader import SecenFlowLoaderMy as DA
        
        train_left_img, train_right_img, train_left_disp, train_left_disp_occ ,test_left_img, test_right_img, test_left_disp, test_left_disp_occ= lt.dataloader(
            args.datapath,data_range_train= 21818,data_range_Val=4248) #21818 4248
        '''
        train_left_img, train_right_img, train_left_disp, train_left_disp_occ ,test_left_img, test_right_img, test_left_disp, test_left_disp_occ = lt.dataloader(
            args.datapath,data_range_train= 7,data_range_Val=3) #200 4248
        '''
    train_left_img.sort()
    train_right_img.sort()
    train_left_disp.sort()
    if args.dataset == "kitti":
        train_left_disp_noc.sort()
        if args.datatype == '2015': 
            train_mask_obj_map.sort()
    else:
        train_left_disp_occ.sort()
    
    test_left_img.sort()
    test_right_img.sort()
    test_left_disp.sort()
    if args.dataset == "kitti":
        test_left_disp_noc.sort()
        if args.datatype == '2015': 
            test_mask_obj_map.sort()
    else:
        test_left_disp_occ.sort()
    
    #__normalize = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    __normalize = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
    if args.dataset == "kitti":
        if args.datatype == '2015':
            TrainImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(train_left_img, train_right_img, train_left_disp,train_left_disp_noc,train_mask_obj_map, True,args.flip_vertical),
                batch_size=args.train_bsize, shuffle=True, num_workers=12, drop_last=False) ## org shuffle = False
            
            #else: #mode= test
            TestImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(test_left_img, test_right_img, test_left_disp,test_left_disp_noc,test_mask_obj_map, False,args.flip_vertical),
                batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)
        else: #args.datatype == '2012':
            TrainImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(train_left_img, train_right_img, train_left_disp,train_left_disp_noc,True,args.flip_vertical),
                batch_size=args.train_bsize, shuffle=True, num_workers=12, drop_last=False) ## org shuffle = False
            #else: #mode= test
            TestImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(test_left_img, test_right_img, test_left_disp,test_left_disp_noc,False,args.flip_vertical),
                batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    else: # args.dataset== "sceneflow"
        #if args.mode == "train":
        TrainImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(train_left_img, train_right_img, train_left_disp,train_left_disp_occ, True, normalize=__normalize),
            batch_size=args.train_bsize, shuffle=True, num_workers=12, drop_last=False)
        #else: #mode= test
        TestImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(test_left_img, test_right_img, test_left_disp,test_left_disp_occ, False, normalize=__normalize),
            batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)#4
    return TrainImgLoader,TestImgLoader

class save_best_checkpoints(object):
    """save best checkpoints"""

    def __init__(self,checkpoints_path,max_checkpoints=10):
        self.best_checkpoints_path = checkpoints_path+"/"+"best_checkpoints"
        CHECK_FOLDER = os.path.isdir(self.best_checkpoints_path)
        if not CHECK_FOLDER:
            os.makedirs(self.best_checkpoints_path)
        self.reset(max_checkpoints)
    
    def reset(self,max_checkpoints):
        self.best_checks=[]
        self.max_checkpoints=max_checkpoints # threshold
        self.counter=0

    def update(self, args, model,return_avg_losses,return_avg_epes,epoch,optimizer,scheduler,minLoss,best_checkpoints):
        if self.counter < self.max_checkpoints:
            sum_losses_all_stages = round(minLoss['sum_losses_stages'],5)
            checkpoint_name = "checkpoint-epoch_"+ str(minLoss['epoch'])+"-sumlosses_"+str(sum_losses_all_stages)+'.pth'
            self.best_checks.append((minLoss,checkpoint_name,sum_losses_all_stages))
            save_chckpoint(args, model,return_avg_losses,return_avg_epes,epoch,optimizer,scheduler,minLoss,self.best_checkpoints_path+"/"+checkpoint_name,best_checkpoints)
            self.counter =self.counter + 1
        else:
            self.best_checks.sort(key=lambda index: index[2])
            remove_check_point = self.best_checks[-1][1]
            check_file_exists = os.path.isfile(os.path.join(self.best_checkpoints_path, remove_check_point))
            if check_file_exists:
                os.remove(os.path.join(self.best_checkpoints_path, remove_check_point))
            self.best_checks = self.best_checks[:-1]
            self.counter =self.counter - 1
            self.update(args, model,return_avg_losses,return_avg_epes,epoch,optimizer,scheduler,minLoss,best_checkpoints)

    def get_best_checkpoint(self):
        self.best_checks.sort(key=lambda index: index[2])
        if len(self.best_checks) > 0:
            return self.best_checks[0]
        else:
            return None


#############
class check_overfit(object):
    """scheck if the training reached overfit"""

    def __init__(self,threshold_overfit_epochs=40):
        self.threshold_overfit_epochs=threshold_overfit_epochs # threshold
        self.reset()
    
    def reset(self):
        self.overfit_started_epoch=-1
        self.counter=0
        self.overfit = 0

    def update(self, args, epoch):
        if self.counter == 0:
            self.overfit_started_epoch = epoch
            self.counter = self.counter + 1
        else:
            if self.counter >= self.threshold_overfit_epochs:
                self.overfit = 1
            else:
                self.counter = self.counter + 1
    
    def get_overfit_status(self):
        return self.overfit, self.overfit_started_epoch

class check_steadyState(object):
    """scheck if the training reached steady state"""

    def __init__(self,threshold_steadystate_epochs=40):
        self.threshold_steadystate_epochs=threshold_steadystate_epochs # threshold
        self.reset()
    
    def reset(self):
        self.steadystate_started_epoch=-1
        self.counter=0
        self.steadystate = 0

    def update(self, args, epoch):
        if self.counter == 0:
            self.steadystate_started_epoch = epoch
            self.counter = self.counter + 1
        else:
            if self.counter >= self.threshold_steadystate_epochs:
                self.steadystate = 1
            else:
                self.counter = self.counter + 1
    
    def get_steadystate_status(self):
        return self.steadystate, self.steadystate_started_epoch

