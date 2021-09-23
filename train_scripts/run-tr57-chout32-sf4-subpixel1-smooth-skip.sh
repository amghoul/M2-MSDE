# -*- coding: UTF-8 -*-
stages=3
dataset=sceneflow # kitti   sceneflow
model=cf_fact3d # org cf_sepconv  cf_fact3d
mode=train # train   finetune    test
datapath=/ds-av/public_datasets/freiburg_sceneflow_subset/raw # /netscratch/alghoul/code/novel-stereonet-v01/FlyingThings3D
datatype=2015
load_numpy=0
flip_vertical=1
epochs=600
lr=0.001
ch_lr_after=20 # FT3d = 20, kitti=200
stepsize=10 # FT3d=10, kitti=50
train_bsize=8 #10
test_bsize=8 #4
save_path=results/tr57-chout32-sf4-subpixel1-smooth-skip
print_freq=50
checkpoint_save_thr=20
abs_thr=3
####load pretrained model
loadmodel=None #/pretrainedModels/org_2Stages_finetune_kitti2015-2020_10_19-21_44_15-epoch-1908-loss1-0.52-lossesSum-1.17-EarlyStopping-stereonet.pth
#### quantization
with_quant=0
quantWL=16 #10
quantFL=8 #8
####for testing
testFile=None #/checkpoint_finetune_kitti2015-2020_12_04-21_28_07-epoch-4000-loss1-0.317-lossesSum-4.391.pth
### for resuming
resume=1 # resume path
resumeFile=/sceneflow_model_cf_fact3d_3stages/checkpoints/best_checkpoints/checkpoint-epoch_20-sumlosses_13.31045.pth
####for model=cf_fact3d
model_bn=1
BN_1D_last=1
BN_1D=1
BN_2D=0
is_filter1_differ=0
filter1_kernels="331 113 113"
fact_kernels="331 113 113"
############# parameters for novel SN
initial_ch=8
initial_scale_factor=4 #4
#num_convs_in_layers="2 2 2" # [2,2,2]
disp_offset=2
sub_pixel_acc=1.0
patch_index=2
is_costvolume_4D=1
chout_costfiltring=32 #0: determinc ch_out in costfiltrting layer accordding to the value of initial_scale_factor
BN_2D_last=0
loss_weights="1.0 1.0 1.0 1.0 1.0"
loss_name="smooth" # gerf smooth
use_skip=1

python3  args_file.py --stages $stages --dataset $dataset --model $model --mode $mode --datapath $datapath \
                                            --datatype $datatype --flip_vertical $flip_vertical --epochs $epochs --lr $lr --train_bsize $train_bsize \
                                            --test_bsize $test_bsize --save_path $save_path --print_freq $print_freq \
                                            --checkpoint_save_thr $checkpoint_save_thr --abs_thr $abs_thr --loadmodel $loadmodel \
                                            --with_quant $with_quant --quantWL $quantWL --quantFL $quantFL --testFile $testFile \
                                            --resume $resume --resumeFile $resumeFile --model_bn $model_bn --BN_1D_last $BN_1D_last \
                                            --BN_1D $BN_1D --BN_2D $BN_2D --is_filter1_differ $is_filter1_differ --filter1_kernels $filter1_kernels \
                                            --fact_kernels $fact_kernels --initial_ch $initial_ch --initial_scale_factor $initial_scale_factor \
                                            --disp_offset $disp_offset --sub_pixel_acc $sub_pixel_acc \
                                            --patch_index $patch_index --is_costvolume_4D $is_costvolume_4D --BN_2D_last $BN_2D_last --chout_costfiltring $chout_costfiltring \
                                            --load_numpy $load_numpy --stepsize $stepsize --ch_lr_after $ch_lr_after --loss_weights $loss_weights --loss_name $loss_name \
                                            --use_skip $use_skip
#CUDA_VISIBLE_DEVICES=0 