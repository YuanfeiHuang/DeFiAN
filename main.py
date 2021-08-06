import os, shutil, time, torch, imageio
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import utils, degradation
from PIL import Image
import model as util_model
import torchvision.transforms as transforms
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import data.common as common
from option import args
from data import data
from flops_counter import get_model_complexity_info
from tqdm import tqdm

def main():
    global opt, model, normalize, unnormalize
    opt = utils.print_args(args)

    # RGB mean for ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    unnormalize = transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444])

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
        if opt.n_GPUs == 1:
            torch.cuda.set_device(opt.GPU_ID)

    cudnn.benchmark = True

    print("===> Building model")
    model = util_model.Generator(opt.n_channels, opt.n_blocks, opt.n_modules, opt.activation,
                                 attention=opt.attention, scale=[opt.degrad['SR_scale']])

    print("===> Calculating NumParams & FLOPs")
    input_size = (3, 480 // opt.degrad['SR_scale'], 360 // opt.degrad['SR_scale'])
    flops, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
    print('-------------Super-resolution Model-------------')
    print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(params * (1e-3), flops * (1e-9), input_size))

    if opt.train:
        start_epoch, model = utils.load_checkpoint(opt.resume, opt.n_GPUs, model, opt.cuda)

        if start_epoch > 0:
            if opt.start_epoch == 0:
                start_epoch = 1

        print("===> Setting GPU")
        if opt.cuda:
            if opt.n_GPUs > 1:
                model = torch.nn.DataParallel(model).cuda()
                para = filter(lambda x: x.requires_grad, model.module.parameters())
            else:
                model = model.cuda()
                para = filter(lambda x: x.requires_grad, model.parameters())
        else:
            para = filter(lambda x: x.requires_grad, model.parameters())

        print("===> Setting Optimizer")
        optimizer = optim.Adam(params=para, lr=opt.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_gamma)

        print("===> Loading Training Dataset")
        train_dataloader = data(opt).get_loader()
        if os.path.exists(opt.model_path + '/' + 'runs'):
            shutil.rmtree(opt.model_path + '/' + 'runs')
        writer = SummaryWriter(opt.model_path + '/runs')

        model.train()

        print("===> Validating")
        for i in range(len(opt.data_test)):
            valid_path = opt.dir_data + 'Test/' + opt.data_test[i]
            validation(valid_path, model, 0)

        for epoch in range(start_epoch, opt.n_epochs + 1):
            print("===> Training on DIV2K-train")
            train(train_dataloader, optimizer, model, epoch, writer)
            utils.save_checkpoint(model, epoch, opt.model_path + '/SR_Model')
            for i in range(len(opt.data_test)):
                valid_path = opt.dir_data + 'Test/' + opt.data_test[i]
                PSNR = validation(valid_path, model, epoch)
                writer.add_scalar(opt.degrad_name + '-PSNR/' + str(opt.data_test[i]), PSNR, epoch)
            torch.cuda.empty_cache()
            scheduler.step()
        writer.close()
    else:
        if opt.preload:
            if opt.n_modules == 10:
                resume = 'checkpoints/' + 'DeFiAN_L_x' + str(opt.degrad['SR_scale']) + '.pth'
            elif opt.n_modules == 5:
                resume = 'checkpoints/' + 'DeFiAN_S_x' + str(opt.degrad['SR_scale']) + '.pth'
            else:
                raise InterruptedError
        else:
            resume = opt.model_path + "/Generator/model_epoch_" + str(opt.best_epoch) + ".pth"
        _, model = utils.load_checkpoint(resume, opt.n_GPUs, model, opt.cuda)

        if opt.cuda:
            if opt.n_GPUs > 1:
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = model.cuda()

        for i in range(len(opt.data_test)):
            valid_path = opt.dir_data + 'Test/' + opt.data_test[i]
            validation(valid_path, model, opt.best_epoch)
        torch.cuda.empty_cache()

def train(training_dataloader, optimizer, model, epoch, writer):
    criterion_MAE = nn.L1Loss(reduction='mean').cuda()
    with tqdm(total=len(training_dataloader)) as pbar:
        for iteration, HR_img in enumerate(training_dataloader):
            HR_img = Variable(HR_img, volatile=False)
            # -----------------------------Validation----------------------------------
            niter = (epoch - 1) * len(training_dataloader) + iteration
            if niter % 200 == 0:
                img_path = opt.dir_data + '/Test/Set5/butterfly.png'
                PSNR = validation_image(img_path, model, niter, writer)
            model.train()
            # ----------------------Preparing degraded LR images------------------------
            prepro = degradation.SRMDPreprocessing(opt.degrad['SR_scale'], random=False,
                                                   kernel=opt.degrad['B_kernel'],
                                                   sig=np.array([opt.degrad['B_sigma']]).repeat(HR_img.shape[0]),
                                                   noise=opt.degrad['N_noise'],
                                                   noise_high=np.array([opt.degrad['N_sigma']]).repeat(HR_img.shape[0]))
            if opt.cuda:
                HR_img = HR_img.cuda()

            LR_img = Variable(normalize(prepro(HR_img)))

            # ----------------------Updating Super-resolution Model (model)------------------------
            optimizer.zero_grad()
            SR_img = model(LR_img)
            loss = criterion_MAE(SR_img, HR_img)
            loss.backward()
            optimizer.step()
            time.sleep(0.01)
            pbar.update(1)

            pbar.set_postfix(Epoch=epoch,
                             LearnRate=optimizer.param_groups[0]["lr"],
                             Loss='%.4f' % loss,
                             PSNR_Img='%.3f' % PSNR,
                             )
            if (niter + 1) % 50 == 0:
                writer.add_scalar('Train/SR_loss', loss, niter)


def validation(valid_path, model, epoch):
    model.eval()
    count = 0
    PSNR = 0

    file = os.listdir(valid_path)
    file.sort()
    length = file.__len__()
    with torch.no_grad():
        with tqdm(total=length) as pbar:
            for idx_img in range(length):
                img_name = file[idx_img].split('.png')[0]
                HR_img = imageio.imread(valid_path + '/' + img_name + '.png')
                HR_img = common.set_channel(HR_img, opt.n_colors)
                HR_img = common.np2Tensor(HR_img, opt.rgb_range)

                HR_img = Variable(HR_img, volatile=False).view(1, HR_img.shape[0], HR_img.shape[1], HR_img.shape[2])

                if opt.cuda:
                    HR_img = HR_img.cuda()

                prepro = degradation.SRMDPreprocessing(opt.degrad['SR_scale'], random=False,
                                                       kernel=opt.degrad['B_kernel'],
                                                       sig=np.array([opt.degrad['B_sigma']]).repeat(HR_img.shape[0]),
                                                       noise=opt.degrad['N_noise'],
                                                       noise_high=np.array([opt.degrad['N_sigma']]).repeat(
                                                           HR_img.shape[0]))
                LR_img = normalize(prepro(HR_img))
                SR_img = model(LR_img)
                SR_img = unnormalize(SR_img)
                if opt.cuda:
                    SR_img = SR_img.data[0].cpu()
                    HR_img = HR_img.data[0].cpu()
                else:
                    SR_img = SR_img.data[0]
                    HR_img = HR_img.data[0]
                PSNR += utils.calc_PSNR(SR_img, HR_img, rgb_range=opt.rgb_range, shave=opt.degrad['SR_scale'])
                count = count + 1

                if not opt.train:
                    SR_path = opt.model_path + '/SRResults/' + valid_path.split('Test/')[1] + '/' + opt.degrad_name
                    if not os.path.exists(SR_path):
                        os.makedirs(SR_path)

                    result = SR_img.mul(255).clamp(0, 255).round()
                    result = result.numpy().astype(np.uint8)
                    result = result.transpose((1, 2, 0))
                    result = Image.fromarray(result)
                    result.save(SR_path + '/' + img_name + '.png')

                Avg_PSNR = PSNR / count
                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(Degrad=opt.degrad_name,
                                 Epoch=epoch,
                                 PSNR='%.4f' % Avg_PSNR)
    torch.cuda.empty_cache()

    return Avg_PSNR

def validation_image(valid_path, model, niter, writer):
    model.eval()
    with torch.no_grad():
        HR_img = imageio.imread(valid_path)
        HR_img = common.set_channel(HR_img, opt.n_colors)
        HR_img = common.np2Tensor(HR_img, opt.rgb_range)
        HR_img = Variable(HR_img, volatile=False).view(1, HR_img.shape[0], HR_img.shape[1], HR_img.shape[2])
        if opt.cuda:
            HR_img = HR_img.cuda()

        prepro = degradation.SRMDPreprocessing(opt.degrad['SR_scale'], random=False,
                                               kernel=opt.degrad['B_kernel'],
                                               sig=np.array([opt.degrad['B_sigma']]).repeat(HR_img.shape[0]),
                                               noise=opt.degrad['N_noise'],
                                               noise_high=np.array([opt.degrad['N_sigma']]).repeat(
                                                   HR_img.shape[0]))
        LR_img = prepro(HR_img)

        SR_img = model(normalize(LR_img))
        SR_img = unnormalize(SR_img)

        if opt.cuda:
            LR_img = LR_img.data[0].cpu()
            SR_img = SR_img.data[0].cpu()
            HR_img = HR_img.data[0].cpu()
        else:
            LR_img = LR_img.data[0]
            SR_img = SR_img.data[0]
            HR_img = HR_img.data[0]
        PSNR = utils.calc_PSNR(SR_img, HR_img, rgb_range=opt.rgb_range, shave=opt.degrad['SR_scale'])
    writer.add_image(opt.degrad_name + '/SR', SR_img.clamp(0, 1).numpy(), niter)
    writer.add_image(opt.degrad_name + '/LR', LR_img.clamp(0, 1).numpy(), niter)
    writer.add_image(opt.degrad_name + '/HR', HR_img.clamp(0, 1).numpy(), niter)
    torch.cuda.empty_cache()

    return PSNR

if __name__ == "__main__":
    main()
