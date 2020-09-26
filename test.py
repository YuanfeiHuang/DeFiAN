import os, cv2

from flops_counter import get_model_complexity_info
import numpy as np
import torch.nn.functional as F
import scipy.misc as misc
import torch, imageio
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import utils
from PIL import Image
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import common, utils
# from Existing_Methods.CARN.carn import Net as Generator
from model import Generator
import matplotlib.pyplot as plt
import argparse
import time
from tqdm import tqdm


def main():
    global model
    parser = argparse.ArgumentParser(description='DeFiAN')
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda?")
    parser.add_argument('--n_GPUs', type=int, default=1, help='parallel training with multiple GPUs')
    parser.add_argument('--GPU_ID', type=int, default=0, help='parallel training with multiple GPUs')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data_scribble loading')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--scale', type=int, default=2, help='scale factor')
    parser.add_argument('--attention', default=True, help='True for DeFiAN')
    parser.add_argument('--n_modules', type=int, default=10, help='num of DeFiAM: 10 for DeFiAN_L; 5 for DeFiAN_S')
    parser.add_argument('--n_blocks', type=int, default=20, help='num of RCABs: 20 for DeFiAN_L; 10 for DeFiAN_S')
    parser.add_argument('--n_channels', type=int, default=64, help='num of channels: 64 for DeFiAN_L; 32 for DeFiAN_S')
    parser.add_argument('--activation', default=nn.ReLU(True), help='activation function')
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        if args.n_GPUs == 1:
            torch.cuda.set_device(args.GPU_ID)
    cudnn.benchmark = True

    model_path = 'checkpoints/'
    if args.n_modules == 5:
        model_path = model_path + 'DeFiAN_S_x' + str(args.scale)
        result_pathes = 'DeFiAN_S/'
    elif args.n_modules == 10:
        model_path = model_path + 'DeFiAN_L_x' + str(args.scale)
        result_pathes = 'DeFiAN_L/'
    else:
        raise InterruptedError

    print("===> Building model")
    model = Generator(args.n_channels, args.n_blocks, args.n_modules, args.activation,
                      attention=args.attention, scale=[args.scale])

    print("===> Calculating NumParams & FLOPs")
    input_size = (3, 480 // args.scale, 360 // args.scale)
    flops, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
    print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(params * (1e-3), flops * (1e-9), input_size))

    cpk = torch.load(model_path + '.pth', map_location={'cuda:1': 'cuda:0'})["state_dict"]
    model.load_state_dict(cpk, strict=False)
    model = model.cuda()

    data_valid = ['Set5_LR_bicubic', 'Urban100_LR_bicubic',
                  'Manga109_LR_bicubic']
    print('====>Testing...')
    for i in range(len(data_valid)):
        result_path = result_pathes + data_valid[i] + '_x' + str(args.scale)
        valid_path = '/mnt/Datasets/Test/' + data_valid[i]
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        valid_psnr, valid_ssim = validation(valid_path, result_path, model, args.scale)
        print('\t {} --- PSNR = {:.4f} SSIM = {:.4f}'.format(data_valid[i], valid_psnr, valid_ssim))


def validation(valid_path, result_path, model, scale):
    model.eval()
    count = 0
    PSNR = 0
    SSIM = 0
    # RGB mean for ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    unnormalize = transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444])

    filepath = valid_path.split('_LR')[0]
    file = os.listdir(filepath)
    lr_file = os.listdir(valid_path + '/X' + str(scale))
    if lr_file[0].find('jpeg') != -1:
        lr_type = '.jpeg'
    elif lr_file[0].find('jpg') != -1:
        lr_type = '.jpg'
    elif lr_file[0].find('bmp') != -1:
        lr_type = '.bmp'
    else:
        lr_type = '.png'

    if file[0].find('jpeg') != -1:
        hr_type = '.jpeg'
    elif file[0].find('jpg') != -1:
        hr_type = '.jpg'
    elif file[0].find('bmp') != -1:
        hr_type = '.bmp'
    else:
        hr_type = '.png'
    file.sort()
    length = file.__len__()
    with torch.no_grad():
        with tqdm(total=length) as pbar:
            for idx_img in range(length):
                time.sleep(0.01)
                pbar.update(1)
                img_name = file[idx_img].split(hr_type)[0]
                img_hr_rgb = imageio.imread(filepath + '/' + img_name + hr_type)
                img_lr_rgb = imageio.imread(
                    valid_path + '/X' + str(scale) + '/' + img_name + 'x' + str(scale) + lr_type)
                img_lr_rgb, img_hr_rgb = common.set_channel(img_lr_rgb, img_hr_rgb, 3)
                img_lr_rgb, img_hr_rgb = common.np2Tensor(img_lr_rgb, img_hr_rgb, 255)

                img_lr_rgb = normalize(img_lr_rgb)  # Normalize
                # img_hr_rgb = normalize(img_hr_rgb)

                img_lr_rgb = Variable(img_lr_rgb).view(1, img_lr_rgb.shape[0], img_lr_rgb.shape[1], img_lr_rgb.shape[2])
                img_hr_rgb = Variable(img_hr_rgb).view(1, img_hr_rgb.shape[0], img_hr_rgb.shape[1], img_hr_rgb.shape[2])

                img_lr_rgb = img_lr_rgb.cuda()
                # SR = F.interpolate(img_lr_rgb, scale_factor=scale)
                SR = model(img_lr_rgb)
                # SR = model(img_lr_rgb, scale)
                SR = unnormalize(SR.data[0].cpu())
                # plt.figure()
                # plt.subplot(1,3, 1)
                # plt.imshow(img_lr_rgb.data[0].cpu().numpy().transpose(1,2,0))
                # plt.subplot(1,3, 2)
                # plt.imshow(SR.numpy().transpose(1,2,0))
                # plt.subplot(1,3, 3)
                # plt.imshow(img_hr_rgb.data[0].cpu().numpy().transpose(1,2,0))
                # plt.show()

                PSNR += utils.calc_PSNR(SR, img_hr_rgb.data[0], rgb_range=255, shave=scale)
                SSIM += utils.calc_SSIM(SR, img_hr_rgb.data[0], rgb_range=255, shave=scale)
                count = count + 1
                result = SR.mul(255).clamp(0, 255).round()
                result = result.numpy().astype(np.uint8)
                result = result.transpose((1, 2, 0))
                result = Image.fromarray(result)
                result.save(result_path + '/' + img_name + '_DeFiAN_x' + str(scale) + '.png')

    Avg_PSNR = PSNR / count
    Avg_SSIM = SSIM / count

    return Avg_PSNR, Avg_SSIM


if __name__ == "__main__":
    main()
