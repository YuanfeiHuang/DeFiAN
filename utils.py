import os, torch, math, time, cv2
import numpy as np
from torch.autograd import Variable
import skimage.color as sc

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

def gram_matrix(input, size_average=False):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    if size_average:
        G = G.div(a * b * c * d)
    return G

def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x

def crop_merge(x, model, scale, shave, min_size, n_GPUs):
    n_GPUs = min(n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [x[:, :, 0:h_size, 0:w_size], x[:, :, 0:h_size, (w - w_size):w],
                 x[:, :, (h - h_size):h, 0:w_size], x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, n_GPUs):
            input_batch = torch.cat(inputlist[i:(i+n_GPUs)], dim=0)
            output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(n_GPUs, dim=0))
    else:
        outputlist = [crop_merge(patch, model, scale, shave, min_size, n_GPUs) for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = Variable(x.data.new(b, c, h, w))
    output[0, :, 0:h_half, 0:w_half] = outputlist[0][0, :, 0:h_half, 0:w_half]
    output[0, :, 0:h_half, w_half:w] = outputlist[1][0, :, 0:h_half, (w_size - w + w_half):w_size]
    output[0, :, h_half:h, 0:w_half] = outputlist[2][0, :, (h_size - h + h_half):h_size, 0:w_half]
    output[0, :, h_half:h, w_half:w] = outputlist[3][0, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def quantize(img, rgb_range):
    return img.mul(255).clamp(0, 255).round().div(255)

def rgb2ycbcrT(rgb):
    rgb = rgb.numpy().transpose(1, 2, 0)
    yCbCr = sc.rgb2ycbcr(rgb) / 255

    return torch.Tensor(yCbCr[:, :, 0])

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_SSIM(input, target, rgb_range, shave):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    c, h, w = input.size()
    if c > 1:
        input = input.mul(255).clamp(0, 255).round()
        target = target[:, 0:h, 0:w].mul(255).clamp(0, 255).round()
        input = rgb2ycbcrT(input)
        target = rgb2ycbcrT(target)
    else:
        input = input
        target = target[:, 0:h, 0:w]
    input = input[shave:(h - shave), shave:(w - shave)]
    target = target[shave:(h - shave), shave:(w - shave)]
    return ssim(input.numpy(), target.numpy())

def calc_PSNR(input, target, rgb_range, shave):

    c, h, w = input.size()
    if c > 1:
        input = quantize(input, rgb_range)
        target = quantize(target[:, 0:h, 0:w], rgb_range)
        input_Y = rgb2ycbcrT(input)
        target_Y = rgb2ycbcrT(target)
        diff = (input_Y - target_Y).view(1, h, w)
    else:
        input = input
        target = target[:, 0:h, 0:w]
        diff = input - target
    diff = diff[:, shave:(h - shave), shave:(w - shave)]
    mse = diff.pow(2).mean()
    psnr = -10 * np.log10(mse)

    return psnr

def weight_svd(module):
    w = module.weight.data
    w = w.view(w.size(0), -1)
    w_sv = np.linalg.svd(w, compute_uv=0)

    return w_sv

def hyperpara_decay(epoch, para, decay, sigma):

    if epoch < decay:
        para = para
    elif epoch < 1.5*decay:
        para = para * sigma
    elif epoch < 2*decay:
        para = para * (sigma**2)
    elif epoch < 2.5*decay:
        para = para * (sigma**3)
    else:
        para = para * (sigma**4)

    return para

def save_checkpoint(model, epoch, folder):

    model_path = folder + "/model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "state_dict": model.state_dict()}
    torch.save(state, model_path)
    print("Checkpoint saved to {}".format(model_path))

def load_checkpoint(resume, n_GPUs, model):
    if os.path.isfile(resume):
        # from collections import OrderedDict
        new_checkpoint = model.state_dict()
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location={'cuda:1':'cuda:0'})
        # checkpoint = torch.load(resume)

        start_epoch = checkpoint["epoch"] + 1
        if n_GPUs > 1:
            for k, v in checkpoint['state_dict'].items():
                if k[:6] != 'module':
                    new_checkpoint[k] = v
                else:
                    name = k[7:]
                    new_checkpoint[name] = v
        else:
            for k, v in checkpoint['state_dict'].items():
                if k[:6] == 'module':
                    name = k[7:]
                    new_checkpoint[name] = v
                    # Change module name when we adjust the model.
                    # if k[:6] == 'module':
                    #     k = 'body_blocks' + k[6:]
                    # new_checkpoint[k] = v
                else:
                    # if k[:7] != 'upscale':
                    #     new_checkpoint[k] = v
                    new_checkpoint[k] = v
        model.load_state_dict(new_checkpoint)
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        start_epoch = 1
    return start_epoch, model

def print_args(args):
    Hess = ''
    if args.attention:
        Name = args.attention + '_H3&H5&H7-OnlyHess|DiEnDec-3to1|LAIN-x|Sigmoid-' + args.SkpConn + '-woTail'
        Hess = '_Hess' + '-G64'
    else:
        Name = args.SkpConn

    args.preload = False

    args.model_path = 'Model-Long_FlickrDIV2K_x' + str(args.scale) + '_' + Name + '_In'+ str(args.patch_size) + '_BS' + str(args.batch_size) + '_lr' + str(args.lr)\
                      + '_B' + str(args.n_blocks) + 'U' + str(args.n_units) + 'F' + str(args.n_feats)\
                      + Hess

    args.resume = args.model_path + '/Generator/model_epoch_' + str(args.start_epoch) +'.pth'
    # args.resume = 'Model_FlickrDIV2K_x2_SRAGE_H3&H5&H7-OnlyHess|DiEnDec-3to1|LAIN-x|Sigmoid-RCAN-woTail_In32_BS16_lr0.0001_B5U10F32_Hess-G64' \
    #               '/Generator/model_epoch_50.pth'

    args.result_path = []
    args.valid_path = []
    for i in range(len(args.data_valid)):
        args.result_path.append(
            args.model_path + "/Results/" + 'Epoch' + str(args.best_epoch) + '/' + args.data_valid[i])
        args.valid_path.append(args.dir_data + "/Test/" + args.data_valid[i] + '_LR_bicubic')
        if not os.path.exists(args.result_path[i]):
            os.makedirs(args.result_path[i])
    if not os.path.exists(args.model_path + '/Generator'):
        os.makedirs(args.model_path + '/Generator')
    print(args)

    return args
