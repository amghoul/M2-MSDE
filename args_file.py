import argparse
from main_file import main as main_function
parser = argparse.ArgumentParser(description='Light StereoNet')
parser.add_argument('--maxdisp', type=int ,default=192, help='maxium disparity')
parser.add_argument('--mode', default='finetune', help='choose mode: train, finetune or test')
parser.add_argument('--loadmodel', default=None, help='load model')
parser.add_argument('--resume', type=int, default=0, help='resme from a specific epock and checkpoint')
parser.add_argument('--resumeFile', type=str, default=None, help='The full path of the reume file')
parser.add_argument('--testFile', type=str, default=None, help='the test file')
parser.add_argument('--datapath', default='/home/alghoul/myenv/kitti2015/training/', help='datapath of the dataset')
parser.add_argument('--load_numpy', type=int, default=0, help='1: load dataset from saved compresse numpy files, amd the datapath set accordingly. 0: load normal dataset')
parser.add_argument('--dataset', type=str, default="kitti", help='Select dataset to work on: sceneflow or kitti')
parser.add_argument('--datatype', default='2015', help='Kitti version 2015 or 2012')
parser.add_argument('--flip_vertical', type=int, default=0, help='flip vertically the KITTI dataset images. 1: flip, 0: no flipping')
parser.add_argument('--save_path', type=str, default='results', help='the path of saving checkpoints and logs for training, fintuning and testing')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--ch_lr_after', type=int, default=5, help='How number of epcohs to use the same value of the initial lr when we start the training ')
parser.add_argument('--train_bsize', type=int, default=1, help='train batch size')
parser.add_argument('--test_bsize', type=int, default=1, help='test batch size')
parser.add_argument('--itersize', default=1, type=int, metavar='IS', help='iter size')
parser.add_argument('--print_freq', type=int, default=1, help='print frequence in log')
parser.add_argument('--with_quant', type=int, default=0, help='finetuning or testing the model with quantization or not')
parser.add_argument('--quantWL', type=int, default=10, help='Number of whole bits in quantization')
parser.add_argument('--quantFL', type=int, default=7, help='Number of float bits in quantization')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--stages', type=int, default=2, help='the stage num of refinement')
parser.add_argument('--abs_thr', type=int, default=3.0, help='absolute error rate threshold')
parser.add_argument('--rel_thr', type=int, default=0.05, help='relative error rate threshold')
parser.add_argument('--number_of_thr', type=int, default=1, help='number of thresholds to use')
parser.add_argument('--model', default='org', help='choose the models: \
    org: original stereonet, \
    cf_sepconv: stereonet with separable conv in cost filtering layer, \
    cf_fact3d: Factorizing Conv3D in cost filtering layer')
parser.add_argument('--BN_1D', type=int, default=1, help='if using Batch Nolrmalized with Conv1D after factorizing Conv3D')
parser.add_argument('--BN_2D', type=int, default=0, help='if using Batch Nolrmalized with Conv2D after factorizing Conv3D')
parser.add_argument('--model_bn', type=int, default=1, help='enabling batch normalies layers in all models or not')
parser.add_argument('--fact_kernels', type=list, nargs='+',default=[[1,1,3],[3,1,1]], help='Kernel values for each conv1d or cov2d layers after factorizing conv3d')
parser.add_argument('--is_filter1_differ', type=int, default=0, help='To use different conv layers in the first component of the cost filtering layer')
parser.add_argument('--filter1_kernels', type=list, nargs='+',default=[[3,1,1],[3,1,1]], help= 'this is for the first component in cost filtering layer. if the above argument is_filter1_differ ==1, then this option is enabled. \
                         Kernel values for each conv1d or cov2d layers after factorizing conv3d')
parser.add_argument('--BN_1D_last', type=int, default=1, help='1: enforce BN for the last conv1D in factorized conv3d regardless of BN_1D and BN_2D values. \
                                                             0: use BN_1D and BN_2D values for BN ')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpu', default='1', type=str, help='GPU ID')
parser.add_argument('--loss_name', default='gerf', help='choose loss name: gerf or smooth')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0, 1.0])
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=1, type=int, metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.6, type=float, help='learning rate decay parameter: Gamma')
parser.add_argument('--checkpoint_save_thr', default=100, type=int, help='How many epochs to save a checkpoint')

parser.add_argument('--initial_ch', default=8, type=int, help='initial channels to be input to the Feature extraction layer')
parser.add_argument('--num_convs_in_layers','--list-type', default=[2,2,2], type=int, nargs='+', help='number of convolutions in each layer scale in the Feature extraction layer')
parser.add_argument('--initial_scale_factor', default=4, type=int, help='initial scale factor to be the output from the Feature extraction layer')
parser.add_argument('--disp_offset', default=2, type=int, help='[-d_ofsset, d_offset] range of desparitesfor the residual and reconstruction cost volumes')
parser.add_argument('--sub_pixel_acc', default=1, type=float, help='sub_pixel_acc is how to incrment d_offset range values either by 1 ot 0.5')
parser.add_argument('--patch_index', default=2, type=float, help='patch_index is the k index of patch size=2k+1')
parser.add_argument('--BN_2D_last', type=int, default=0, help='1: This is apply if the generated cost volume is 3D (D,H,W) if BN_2D_last = 1 and model_BN = 1 then use BN in the last Conv2D layer,  \
                                                                else if BN_2D_last=0 and model_BN = 1 then use BN in all conv2D costfiltering layers .')
parser.add_argument('--is_costvolume_4D', type=int, default=0, help='1: use cost volume 4D with dimension (B,1,D,H,W), if 0 then use 3D cost volume (B,D,H,W).')
parser.add_argument('--chout_costfiltring', type=int, default=0, help='1: if <=0, then use the initial_scale_factor value to determine ch_out in cost filtering layer, if >0, use this value in ch_out')

parser.add_argument('--max_checkpoints_to_save', default=10, type=int, help='How many best checkpoints to save')
parser.add_argument('--threshold_overfit_epochs', default=50, type=int, help='number of epochs in which training loss strat increasing to consider the model was reacched to overfit')
parser.add_argument('--use_skip', default=0, type=int, help='1: use skip conections in filter blocks. 0: no skip connections are used')
#############

args = parser.parse_args() # uncomment colab

if __name__ == '__main__':
    main_function(args)