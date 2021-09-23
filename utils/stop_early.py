import copy
from .save_load import *

def stop_early(args,model,optimizer,scheduler,log,current_val_error,best_error,timestr,epoch,return_avg_losses,
    return_sum_stages_losses,minLoss,earlyStoppedEpochs,n_stop_epochs,no_improvement_epochs,best_model,ealryStoppedPaths,best_model_dict,checkpoints_path,best_checkpoints,return_avg_epes):
    
    if current_val_error < best_error :
        best_error = current_val_error
        best_model = copy.deepcopy(model)
        checkName = args.model+'_fin_'+args.dataset+'-'+args.datatype+'-'+timestr+'-epoch-'+str(epoch)+'-loss'+str(args.stages-1)+'-'+str(round(return_avg_losses[args.stages-1],3))+'-lossesSum-'+str(round(return_sum_stages_losses,3))+'-EarlyStopping'+'.pth'
        savefilenameEarlyStopping = checkpoints_path + '/' + checkName
        
        best_model = copy.deepcopy(model)
        best_model_dict['temp_return_avg_losses']=return_avg_losses
        best_model_dict['temp_optimizer_state_dict']=optimizer
        best_model_dict['sheduler']=scheduler
        best_model_dict['temp_min_loss']=minLoss
        best_model_dict['temp_epoch']=epoch
        best_model_dict['temp_savefilenameEarlyStopping']=savefilenameEarlyStopping
        
        no_improvement_epochs = 0
        ealryStoppedPaths.append(savefilenameEarlyStopping)
        earlyStoppedEpochs.append(epoch)
    else:
        no_improvement_epochs += 1
        log.info("no_improvement_epochs "+ str(no_improvement_epochs) +" current_epoch "+str(epoch))
        if no_improvement_epochs == n_stop_epochs :
            log.info('+++++++++++++++++++ Early Stopping  started below at epoch: '+ str(epoch)+'++++++++++++++++')
            checkName = args.model+'_fin_'+args.dataset+'-'+args.datatype+'-'+timestr+'-epoch-'+str(epoch)+'-loss'+str(args.stages-1)+'-'+str(round(return_avg_losses[args.stages-1],3))+'-lossesSum-'+str(round(return_sum_stages_losses,3))+'-EarlyStopping-at-epoch-'+str(earlyStoppedEpochs[-1])+'.pth'
            savefilename = checkpoints_path + '/' + checkName
            savefilenameEarlyStopping = ealryStoppedPaths[-1]
            
            temp_avg_train_loss_stage=best_model_dict['temp_return_avg_losses']
            temp_epoch=best_model_dict['temp_epoch']
            temp_optimizer_state_dict=best_model_dict['temp_optimizer_state_dict']
            temp_sheduler=best_model_dict['sheduler']
            temp_best_losses_stages=best_model_dict['temp_min_loss']

            save_chckpoint(args, best_model,temp_avg_train_loss_stage,return_avg_epes,temp_epoch,temp_optimizer_state_dict,temp_sheduler,temp_best_losses_stages,savefilenameEarlyStopping,best_checkpoints)

            log.info('+++++++++++++++++++ saved early stopping model ended above at epoch: '+ str(epoch)+' ++++++++++++++++')
    return best_model,no_improvement_epochs,best_model_dict,ealryStoppedPaths,best_error,earlyStoppedEpochs

def get_avgErr_SPlitEpochs(avgErr_for_split_epochs,minErr_for_split_epochs,GL_for_split_epochs,sumErr,split_epochs,temp_minErr):
    avgErr_for_split_epochs.append(sumErr/split_epochs)
    minErr_for_split_epochs.append(temp_minErr)
    GL = ((avgErr_for_split_epochs[-1] / minErr_for_split_epochs[-1])-1)*100
    GL_for_split_epochs.append(GL)
    temp_minErr=float('inf')
    sumErr= 0
    return avgErr_for_split_epochs,minErr_for_split_epochs,GL_for_split_epochs,temp_minErr,sumErr