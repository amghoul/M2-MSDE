import os
import sys
import time
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np

from qtorch import FixedPoint, FloatingPoint

from models.StereoNet_Multi import StereoNet
from models.StereoNet_Multi_SepConv import StereoNet as StereoNetSepConv
from models.StereoNet_Multi_FactorizedConv3D import StereoNet as StereoNetFactorized
from save_dataset_numpy import load_saved_numpy_dataset

from utils.folder_logs_init import *
from utils.save_load import (
    load_dataset, save_best_lossess, save_chckpoint, load_checkpoint,
    load_quantized_model, save_losses, save_GL,
)
from utils.test import test
from utils.train import *
from utils.stop_early import *

from dataclasses_models import (
    ModelBundle, Loaders, Paths, MetricsHistory, GeneralizationTracker, EarlyStopping,
)


# ----------------------------------------------------------------------------
# Model construction.
# ----------------------------------------------------------------------------
def _build_network(args):
    """Construct the right StereoNet variant. No logging, no device placement."""
    if args.model == 'cf_sepconv':
        return StereoNetSepConv(k=3, r=args.stages - 1, maxdisp=args.maxdisp)
    elif args.model == 'cf_fact3d':
        return StereoNetFactorized(
            r=args.stages - 1, use_skip=args.use_skip, initial_ch=args.initial_ch,
            num_convs_in_layers=args.num_convs_in_layers, initial_scale_factor=args.initial_scale_factor,
            disp_offset=args.disp_offset, sub_pixel_acc=args.sub_pixel_acc, patch_index=args.patch_index,
            is_costvolume_4D=args.is_costvolume_4D, BN_2D_last=args.BN_2D_last,
            chout_costfiltring=args.chout_costfiltring, is_filter1_differ=args.is_filter1_differ,
            filter1_kernels=args.filter1_kernels, fact_kernels=args.fact_kernels,
            BN_1D=args.BN_1D, BN_2D=args.BN_2D, BN_1D_last=args.BN_1D_last,
            model_bn=args.model_bn, maxdisp=args.maxdisp)
    else:
        return StereoNet(k=3, r=args.stages - 1, maxdisp=args.maxdisp)


def _to_device(args, model):
    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()
    return model


def _build_for_train(args, log):
    if args.model == 'cf_sepconv':
        log.info("-- model using stereonet with Separable Covolution for training--")
    elif args.model == 'cf_fact3d':
        log.info("-- model using stereonet with Factorized Conv3D for Cost Filtering Layer for training--")
    else:
        log.info("-- model using Original StereoNet for training--")

    model = _to_device(args, _build_network(args))

    if args.dataset == "kitti":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif args.dataset == "sceneflow":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        raise ValueError("Unsupported dataset for train mode: {}".format(args.dataset))

    return ModelBundle(model, optimizer, scheduler)


def _build_for_finetune(args, log, root_path):
    pretrained_path = root_path + str(args.loadmodel)
    if not os.path.isfile(pretrained_path):
        log.info("No ptratrained model exist in the path: " + pretrained_path)
        sys.exit()

    if args.model == 'cf_sepconv':
        log.info("-- model using stereonet with Separable Covolution for finetuning--")
        model = _to_device(args, _build_network(args))
        log.info("-- pretrained model loaded for separable convolution for cost filtering layer --")
    elif args.model == 'cf_fact3d':
        log.info("-- model using stereonet with factorized conv3d in the cost vlume filtering for finetuning--")
        model = _to_device(args, _build_network(args))
    else:  # original stereonet
        log.info("-- model using the original stereonet --")
        model = _to_device(args, _build_network(args))
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict['state_dict'])
        log.info("-- pretrained model was loaded for the original StereoNet --")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    return ModelBundle(model, optimizer, scheduler)


def _build_for_test(args, log):
    if args.model == 'cf_sepconv':
        log.info("-- model using stereonet with Separable Covolution for testing--")
    elif args.model == 'cf_fact3d':
        log.info("-- model using stereonet with Factorized Conv3D for Cost Filtering Layer for testing--")
    else:
        log.info("-- model using Original StereoNet for testing--")

    model = _to_device(args, _build_network(args))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    return ModelBundle(model, optimizer, scheduler)


def build_model(args, log, root_path):
    if args.mode == "train":
        return _build_for_train(args, log)
    elif args.mode == "finetune":
        return _build_for_finetune(args, log, root_path)
    elif args.mode == "test":
        return _build_for_test(args, log)
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))


# ----------------------------------------------------------------------------
# Setup.
# ----------------------------------------------------------------------------
def prepare_args(args):
    args.stages = len(args.num_convs_in_layers)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    for i in range(len(args.fact_kernels)):
        args.fact_kernels[i] = list(map(int, args.fact_kernels[i]))
    for i in range(len(args.filter1_kernels)):
        args.filter1_kernels[i] = list(map(int, args.filter1_kernels[i]))


def load_loaders(args):
    if args.load_numpy == 1:
        train_loader = load_saved_numpy_dataset(
            args.datapath + "/train/", batch_size=args.train_bsize, shuffle=True, num_workers=12)
        test_loader = load_saved_numpy_dataset(
            args.datapath + "/val/", batch_size=args.test_bsize, shuffle=False, num_workers=4)
    else:
        train_loader, test_loader = load_dataset(args)
    return Loaders(train=train_loader, test=test_loader)


def setup_experiment(args):
    stages = args.stages

    n_stop_epochs = 25  # for early stopping
    if args.dataset == "sceneflow":
        n_stop_epochs = 5

    split_epochs = 20
    if args.dataset == "sceneflow":
        split_epochs = 5

    minLoss = {
        'loss0': float('inf'), 'loss1': float('inf'), 'loss2': float('inf'),
        'loss3': float('inf'), 'sum_losses_stages': float('inf'),
        'test_sum_losses_stages': float('inf'), 'checkpointPath': 'none', 'epoch': 1,
    }

    paths = Paths(*inialize_log_file(args))

    loaders = load_loaders(args)

    bundle = build_model(args, paths.log, paths.root)

    max_checkpoints_to_save = args.max_checkpoints_to_save
    threshold_overfit_epochs = args.threshold_overfit_epochs
    best_checkpoints = save_best_checkpoints(paths.checkpoints, max_checkpoints_to_save)
    check_train_overfit = check_overfit(threshold_overfit_epochs)
    threshold_steadystate_epochs = threshold_overfit_epochs
    check_train_steadystate = check_steadyState(threshold_steadystate_epochs)

    paths.log.info("GPU is: " + torch.cuda.get_device_name(torch.cuda.current_device()))
    paths.log.info('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in bundle.model.parameters()])))
    paths.log.info('Number of images to train is: {} with train batch size: {}'.format(
        len(loaders.train), args.train_bsize))
    paths.log.info('Number of images to test is: {} with test batch size: {}'.format(
        len(loaders.test), args.test_bsize))

    history = MetricsHistory()
    gl = GeneralizationTracker(split_epochs=split_epochs)
    early = EarlyStopping(
        overfit_checker=check_train_overfit,
        steadystate_checker=check_train_steadystate,
        best_checkpoints=best_checkpoints,
        n_stop_epochs=n_stop_epochs,
        threshold_epochs=threshold_overfit_epochs,
    )

    return bundle, loaders, paths, history, gl, early, minLoss, stages


# ----------------------------------------------------------------------------
# Per-epoch work.
# ----------------------------------------------------------------------------
def train_one_epoch(args, bundle, loaders, paths, history, float_model_dict, epoch):
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    paths.log.info('############## This is %d-th epoch' % epoch + " ###############")

    epoch_train_start_time = time.time()
    if args.with_quant == 1:
        (return_avg_losses, return_sum_stages_losses, float_model_dict,
         return_avg_epes, return_sum_stages_epes) = train(
            args, loaders.train, bundle.model, bundle.optimizer, paths.log, float_model_dict, epoch)
    else:
        (return_avg_losses, return_sum_stages_losses, float_model_dict,
         return_avg_epes, return_sum_stages_epes) = train(
            args, loaders.train, bundle.model, bundle.optimizer, paths.log, None, epoch)
    epoch_train_end_time = time.time()

    history.losses_all_stages[epoch] = return_avg_losses
    history.losses_sum_all_stages[epoch] = return_sum_stages_losses
    history.epes_all_stages[epoch] = return_avg_epes
    history.epes_sum_all_stages[epoch] = return_sum_stages_epes

    if args.with_quant == 1:
        check_name = (args.model + '_fin_withQuant_' + args.dataset + '-' + args.datatype + '-' + timestr
                      + '-epoch-' + str(epoch) + '-loss' + str(args.stages - 1) + '-'
                      + str(round(return_avg_losses[args.stages - 1], 3)) + '-lossesSum-'
                      + str(round(return_sum_stages_losses, 3)) + '.pth')
    else:
        check_name = (args.model + '_fin_' + args.dataset + '-' + args.datatype + '-' + timestr
                      + '-epoch-' + str(epoch) + '-loss' + str(args.stages - 1) + '-'
                      + str(round(return_avg_losses[args.stages - 1], 3)) + '-lossesSum-'
                      + str(round(return_sum_stages_losses, 3)) + '.pth')

    bundle.scheduler, bundle.optimizer = adjust_learning_rate(
        args, bundle.scheduler, bundle.optimizer, epoch, paths.log)

    return {
        'timestr': timestr,
        'check_name': check_name,
        'return_avg_losses': return_avg_losses,
        'return_sum_stages_losses': return_sum_stages_losses,
        'return_avg_epes': return_avg_epes,
        'return_sum_stages_epes': return_sum_stages_epes,
        'float_model_dict': float_model_dict,
        'epoch_train_start_time': epoch_train_start_time,
        'epoch_train_end_time': epoch_train_end_time,
    }


def validate_epoch(args, bundle, loaders, paths, history, epoch):
    paths.log.info('##### validation for epoch ' + str(epoch) + '#####')
    epoch_test_start_time = time.time()
    (test_return_avg_losses, test_return_avg_EPEs, mask_names, return_test_sum_stages_losses,
     return_outliers_sumary1, return_outliers_sumary2, return_outliers_sumary3) = test_from_training(
        args, loaders.test, bundle.model, paths.log)
    epoch_test_end_time = time.time()

    history.test_losses[epoch] = test_return_avg_losses
    history.test_epes[epoch] = test_return_avg_EPEs
    history.test_sum_stages_losses[epoch] = return_test_sum_stages_losses
    history.test_outliers_1[epoch] = return_outliers_sumary1
    history.test_outliers_2[epoch] = return_outliers_sumary2
    history.test_outliers_3[epoch] = return_outliers_sumary3

    return {
        'mask_names': mask_names,
        'return_test_sum_stages_losses': return_test_sum_stages_losses,
        'epoch_test_start_time': epoch_test_start_time,
        'epoch_test_end_time': epoch_test_end_time,
    }


# ----------------------------------------------------------------------------
# Training loop.
# ----------------------------------------------------------------------------
def run_training(args, bundle, loaders, paths, history, gl, early, minLoss, stages):
    paths.log.info("model loaded with k= 3 " + "and stages= " + str(args.stages))

    prev_return_avg_losses = []
    return_avg_losses = []
    for x in range(stages):
        prev_return_avg_losses.append(float('inf'))
        return_avg_losses = [float('inf')]  # NOTE: preserved exactly as in the original (see message)

    prev_sum_stages_losses = float('inf')
    epoch_start = 1
    float_model_dict = None

    # resume
    if args.resume == 1:
        resumeFile = paths.root + "/" + args.save_path + args.resumeFile
        if os.path.isfile(resumeFile):
            (bundle.model, return_avg_losses, ret_avg_train_epe_stage, bundle.optimizer,
             bundle.scheduler, epoch_start, current_lr, minLoss, saved_args,
             other_returned_values) = load_checkpoint(
                resumeFile, bundle.model, bundle.optimizer, bundle.scheduler)
            if len(other_returned_values) != 0:
                early.best_checkpoints = other_returned_values[0]
            paths.log.info("-- checkpoint loaded and the training is resumed from epoch " + str(epoch_start) + " --")
        else:
            paths.log.info("The resume file is: " + resumeFile)
            paths.log.info("-----The resume file is not exist------")
            sys.exit()

    if args.with_quant == 1:
        bundle.model, float_model_dict = load_quantized_model(args, bundle.model)
    paths.log.info(bundle.model)

    if args.resume == 1:
        paths.log.info("*********The_resumed_learning_rate: {:.12f} ".format(bundle.optimizer.param_groups[0]['lr']))
    else:
        paths.log.info("*********The_initial_learning_rate: {:.12f} ".format(bundle.optimizer.param_groups[0]['lr']))

    start_full_time = time.time()

    if args.epochs <= epoch_start:
        if args.resume == 1:
            paths.log.info("-----The model will resume from epoch " + str(epoch_start) + ", so epochs value " + str(args.epochs) + " in argss file must be larger than " + str(epoch_start) + " ------")
            sys.exit()
        else:
            paths.log.info("-----the model will start training or finetuning from epoch " + str(epoch_start) + ", so epochs value " + str(args.epochs) + " in argss file must be larger than " + str(epoch_start) + " ------")
            sys.exit()

    for epoch in range(epoch_start, args.epochs + 1):
        is_steadystate, steadystate_started_epoch = early.steadystate_checker.get_steadystate_status()
        is_overfit, overfit_started_epoch = early.overfit_checker.get_overfit_status()

        if is_overfit == 1:
            paths.log.info("!!!!!!!!!!! We reached overfit for " + str(early.threshold_epochs) + " epochs. Overfit started at epoch: " + str(overfit_started_epoch))
            break
        elif is_steadystate == 1:
            paths.log.info("!!!!!!!!!!! We reached steady state for " + str(early.threshold_epochs) + " epochs. steady state started at epoch: " + str(steadystate_started_epoch))
            break
        else:  # no overfit or steady state
            tr = train_one_epoch(args, bundle, loaders, paths, history, float_model_dict, epoch)
            timestr = tr['timestr']
            check_name = tr['check_name']
            return_avg_losses = tr['return_avg_losses']
            return_sum_stages_losses = tr['return_sum_stages_losses']
            return_avg_epes = tr['return_avg_epes']
            return_sum_stages_epes = tr['return_sum_stages_epes']
            float_model_dict = tr['float_model_dict']
            epoch_train_start_time = tr['epoch_train_start_time']
            epoch_train_end_time = tr['epoch_train_end_time']

            va = validate_epoch(args, bundle, loaders, paths, history, epoch)
            mask_names = va['mask_names']
            return_test_sum_stages_losses = va['return_test_sum_stages_losses']
            epoch_test_start_time = va['epoch_test_start_time']
            epoch_test_end_time = va['epoch_test_end_time']

            save_losses(
                mask_names, bundle.optimizer, paths.losses_file, epoch, stages,
                history.losses_all_stages, history.epes_all_stages, history.test_losses,
                history.test_epes, history.losses_sum_all_stages, history.epes_sum_all_stages,
                history.test_sum_stages_losses, history.test_outliers_1, history.test_outliers_2,
                history.test_outliers_3, epoch_train_end_time, epoch_train_start_time,
                epoch_test_end_time, epoch_test_start_time, bundle.scheduler)

            savefilename = paths.checkpoints + '/' + check_name
            minLoss = save_best_lossess(
                args, return_avg_losses, return_sum_stages_losses, return_avg_epes,
                return_sum_stages_epes, return_test_sum_stages_losses[0], minLoss, epoch,
                savefilename, bundle.model, bundle.optimizer, bundle.scheduler,
                early.best_checkpoints, early.overfit_checker)

            if epoch % args.checkpoint_save_thr == 0:  # 100
                print(savefilename)
                save_chckpoint(args, bundle.model, return_avg_losses, return_avg_epes, epoch,
                               bundle.optimizer, bundle.scheduler, minLoss, savefilename, early.best_checkpoints)

            # check if reached overfit or steady state
            reach_overfit_or_steadystate(
                args, prev_sum_stages_losses, return_sum_stages_losses, epoch,
                early.overfit_checker, early.steadystate_checker)

            prev_return_avg_losses = return_avg_losses[:]
            prev_sum_stages_losses = return_sum_stages_losses

            # generalization training/testing error
            is_split = gl.update(
                history.losses_sum_all_stages[epoch],
                history.test_sum_stages_losses[epoch][0],
                epoch)

            if is_split:
                save_GL(paths.gl_file, gl.avg_tr[-1], gl.min_tr[-1], gl.avg_val[-1],
                        gl.min_val[-1], gl.gl_tr[-1], gl.gl_val[-1])

                # early stopping
                best_model_dict = {}
                best_model_dict['temp_return_avg_losses'] = return_avg_losses
                best_model_dict['temp_optimizer_state_dict'] = bundle.optimizer
                best_model_dict['sheduler'] = bundle.scheduler
                best_model_dict['temp_min_loss'] = minLoss
                best_model_dict['temp_epoch'] = epoch
                best_model_dict['temp_savefilenameEarlyStopping'] = 'none'
                best_model = copy.deepcopy(bundle.model)

                (best_model, early.no_improvement_epochs, best_model_dict, early.stopped_paths,
                 early.best_error, early.stopped_epochs) = stop_early(
                    args, bundle.model, bundle.optimizer, bundle.scheduler, paths.log,
                    gl.avg_val[-1], early.best_error, timestr, epoch, return_avg_losses,
                    return_sum_stages_losses, minLoss, early.stopped_epochs, early.n_stop_epochs,
                    early.no_improvement_epochs, best_model, early.stopped_paths,
                    best_model_dict, paths.checkpoints, early.best_checkpoints, return_avg_epes)

    if args.with_quant == 1:
        model_save_path = (paths.checkpoints + '/' + 'checkpoint_withQuantize_finetune_kitti'
                           + args.datatype + '-' + timestr + '-epoch-' + str(epoch) + '-loss'
                           + str(args.stages - 1) + '-' + str(round(return_avg_losses[args.stages - 1], 3))
                           + '-lossesSum-' + str(round(return_sum_stages_losses, 3)) + '.pth')
        save_chckpoint(args, bundle.model, return_avg_losses, return_avg_epes, epoch,
                       bundle.optimizer, bundle.scheduler, minLoss, model_save_path, early.best_checkpoints)

    if early.best_checkpoints.get_best_checkpoint() is not None:
        paths.log.info('#################The best checkpoint is: minLoss, best_path, best_sum_stages_losses ##########')
        min_loss_log = ""
        for k, v in early.best_checkpoints.get_best_checkpoint()[0].items():
            if k == 'checkpointPath' or k == 'epoch':
                min_loss_log = min_loss_log + ', '.join([k + ": " + str(v) + ", "])
            else:
                min_loss_log = min_loss_log + ', '.join([k + ": " + str(round(v, 4)) + ", "])
        paths.log.info('Min losses are: ' + min_loss_log)
        paths.log.info("The name of the best checkpoint is: " + early.best_checkpoints.get_best_checkpoint()[1])
        paths.log.info("The minimum sum stages loss is: " + str(early.best_checkpoints.get_best_checkpoint()[2]))
    else:
        paths.log.info('#################The best checkpoint is empty ##########')
    paths.log.info('full_training_time: {: 3f} Hours'.format((time.time() - start_full_time) / 3600))


# ----------------------------------------------------------------------------
# Testing.
# ----------------------------------------------------------------------------
def run_test(args, bundle, loaders, paths, stages):
    infor_str = ''.join(['############# Start testing'])
    paths.log.info(infor_str)
    testFile = paths.root + "/" + args.save_path + args.testFile
    if os.path.isfile(testFile):
        # loading saved values in the checkpoint
        if args.with_quant == 1:
            (bundle.model, t_avg_train_loss_stages, avg_train_epe_stage, bundle.optimizer,
             bundle.scheduler, t_epoch, current_lr, temp_best_losses_stages, saved_args,
             other_returned_values) = load_checkpoint(testFile, bundle.model, bundle.optimizer, bundle.scheduler)
            if len(other_returned_values) != 0:
                best_checkpoints = other_returned_values[0]

            if saved_args.with_quant == 0:  # if the saved model was not quantized previously
                forward_num = FixedPoint(wl=args.quantWL, fl=args.quantFL, clamp=True, symmetric=False)
                backward_num = FloatingPoint(exp=4, man=4)
                layerTypes = ['activation', 'loss']
                bundle.model = sequential_lower(bundle.model, layer_types=layerTypes, forward_number=forward_num)
                quant_model_dict = quantize_model(bundle.model.state_dict(), args.quantWL, args.quantWL - args.quantFL)
                bundle.model.load_state_dict(quant_model_dict)
        else:
            (bundle.model, t_avg_train_loss_stages, t_avg_train_epe_stage, bundle.optimizer,
             bundle.scheduler, t_epoch, current_lr, temp_best_losses_stages, saved_args,
             other_returned_values) = load_checkpoint(testFile, bundle.model, bundle.optimizer, bundle.scheduler)
            if len(other_returned_values) != 0:
                best_checkpoints = other_returned_values[0]

        if t_avg_train_epe_stage is not None:
            paths.log.info('Testing model at epoch {} with: sum_losses_stages is {:.3f} and sum_EPEs_stages is {:.3f}, current_learning_rate {:.10f} '.format(
                t_epoch, sum(t_avg_train_loss_stages[x] for x in range(len(t_avg_train_loss_stages))),
                sum(t_avg_train_epe_stage[x] for x in range(len(t_avg_train_epe_stage))), current_lr))
        else:
            paths.log.info('Testing model at epoch {} with: sum_losses_stages is {:.3f} and current_learning_rate {:.10f} '.format(
                t_epoch, sum(t_avg_train_loss_stages[x] for x in range(len(t_avg_train_loss_stages))), current_lr))

        info_str3 = ', '.join(['Stage{} {:.3f}'.format(x, temp_best_losses_stages['loss' + str(x)]) for x in range(stages)])
        paths.log.info('The best loss is for epoch' + str(temp_best_losses_stages['epoch']) + ' with losses: ' + info_str3)
        paths.log.info('The sum of validation losses of this best checkpoint is: {:.5f}'.format(temp_best_losses_stages['test_sum_losses_stages']))
        paths.log.info("*************** Saved Args in Checkpoint **************")
        paths.log.info("The saved args in checkpoint are:")
        for key, value in sorted(vars(saved_args).items()):
            paths.log.info(str(key) + ':' + str(value))
        paths.log.info("*******************************************************")

        start_test_full_time = time.time()
        outliers_summary1, outliers_summary2, outliers_summary3, mask_names = test(
            args, loaders.test, bundle.model, paths.test_results, paths.log, paths.valid_results)
        testing_time = time.time() - start_test_full_time
        paths.log.info('The testing time is %d sec (%.3f min)' % (testing_time, testing_time / 60))
        paths.log.info('The result path is: ' + paths.test_results)
    else:
        paths.log.info("There is no checkpoint to test")
        sys.exit()


# ----------------------------------------------------------------------------
# Entry point.
# ----------------------------------------------------------------------------
def main(args):
    prepare_args(args)
    bundle, loaders, paths, history, gl, early, minLoss, stages = setup_experiment(args)

    if args.mode == "finetune" or args.mode == "train":
        run_training(args, bundle, loaders, paths, history, gl, early, minLoss, stages)
    else:  # test
        run_test(args, bundle, loaders, paths, stages)


if __name__ == '__main__':
    main()
