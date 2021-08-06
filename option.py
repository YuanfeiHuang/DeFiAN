import argparse
import torch.nn as nn

parser = argparse.ArgumentParser(description='DeFiAN')

# Hardware specifications
parser.add_argument("--cuda", default=False, action="store_true", help="Use cuda?")
parser.add_argument('--n_GPUs', type=int, default=1, help='parallel training with multiple GPUs')
parser.add_argument('--GPU_ID', type=int, default=0, help='GPU ID')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# data specifications
parser.add_argument('--dir_data', type=str, default='../Datasets/', help='dataset directory')
parser.add_argument('--ext', type=str, default='img', help='dataset file extension')
parser.add_argument('--data_train', type=str, default='DIV2K', help='training dataset name')
parser.add_argument('--data_test', type=str, default=['Set5'], help='validation/test dataset')
parser.add_argument('--n_train', type=int, default=800, help='number of training set')
parser.add_argument('--degrad', type=float, default={'SR_scale': 2,
                                                     'B_kernel': False, 'B_sigma': 0,
                                                     'N_noise': False, 'N_sigma': 0}, help='degradation prior')
parser.add_argument('--patch_size', type=int, default=128, help='output patch size for training')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--chop_forward', action='store_true', help='enable memory-efficient forward')

# Model specifications:
parser.add_argument('--attention', default=True, action="store_true", help='number of ResNet units')
parser.add_argument('--n_modules', type=int, default=5, help='number of DeFiAM: N in the paper')
parser.add_argument('--n_blocks', type=int, default=10, help='number of blocks in each FEM: M in the paper')
parser.add_argument('--n_channels', type=int, default=32, help='number of channels: C in the paper')
parser.add_argument('--activation', default=nn.ReLU(inplace=True), help='activation function')

# Training specifications
parser.add_argument("--train", default=True, action="store_true", help="True for training, False for testing")
parser.add_argument("--preload", default=True, action="store_true", help="Pre-load the pretrained checkpoints")
parser.add_argument('--iter_epoch', type=int, default=2000, help='iteration in each epoch')
parser.add_argument("--start_epoch", default=0, type=int, help="start epoch for training")
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--best_epoch', type=int, default=300, help='best epoch from validation PSNR')
parser.add_argument('--resume', type=str, default='', help='load the model from the specified epoch')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--min_size',type=int, default=160000, help='limited size for GPUs processing')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for training SR_Model')
parser.add_argument('--lr_step_size', type=int, default=100, help='learning rate decay per N epochs')
parser.add_argument('--lr_gamma', type=int, default=0.5, help='learning rate decay factor for step decay')

args = parser.parse_args()
