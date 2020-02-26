"""
Code for "Invertible Residual Networks"
http://proceedings.mlr.press/v97/behrmann19a.html
ICML, 2019
"""

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import torchvision.transforms as transforms
#import visdom
import os
import sys
import time
import argparse
import pdb
import random
import json
from models.utils_cifar import train, test, std, mean, get_hms, interpolate
from models.conv_iResNet import conv_iResNet as iResNet
from models.conv_iResNet import multiscale_conv_iResNet as multiscale_iResNet
from models.ucf101_loader import UCF101
import traceback

parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on Cifar')
parser.add_argument('-densityEstimation', '--densityEstimation', dest='densityEstimation',
                    action='store_true', help='perform density estimation')
parser.add_argument('--optimizer', type=str, help="optimizer",
                    choices=["adam", "adamax", "sgd"],
                    default="sgd")#"adamax"
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--coeff', default=0.9, type=float, help='contraction coefficient for linear layers')
parser.add_argument('--numTraceSamples', default=1, type=int, help='number of samples used for trace estimation')
parser.add_argument('--numSeriesTerms', default=1, type=int, help='number of terms used in power series for matrix log')
parser.add_argument('--powerIterSpectralNorm', type=int,
                    help='number of power iterations used for spectral norm',
                    default=1)#5
parser.add_argument('--weight_decay', default=5e-4, type=float, help='coefficient for weight decay')
parser.add_argument('--drop_rate', default=0.1, type=float, help='dropout rate')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--init_batch', default=1024, type=int, help='init batch size')
parser.add_argument('--init_ds', type=int, help='initial downsampling',
                    default=1)#2
parser.add_argument('--warmup_epochs', default=10, type=int, help='epochs for warmup')
parser.add_argument('--inj_pad', type=int, help='initial inj padding',
                    default=13)#0
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--nBlocks', nargs='+', type=int,
                    default=[7, 7, 7])#default=[4, 4, 4])
parser.add_argument('--nStrides', nargs='+', type=int, default=[1, 2, 2])
parser.add_argument('--nChannels', nargs='+', type=int,
                    default=[32, 64, 128]) # default=[16, 64, 256])
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-interpolate', '--interpolate', dest='interpolate', action='store_true', help='train iresnet')
parser.add_argument('-drop_two', '--drop_two', dest='drop_two', action='store_true', help='2d dropout on')
parser.add_argument('-nesterov', '--nesterov', dest='nesterov', action='store_true',
                    help='nesterov momentum')
parser.add_argument('-norm', '--norm', dest='norm', action='store_true',
                    help='compute norms of conv operators')
parser.add_argument('-analysisTraceEst', '--analysisTraceEst', dest='analysisTraceEst', action='store_true',
                    help='analysis of trace estimation')
parser.add_argument('-multiScale', '--multiScale', dest='multiScale', action='store_true',
                    help='use multiscale')
parser.add_argument('-fixedPrior', '--fixedPrior', dest='fixedPrior', action='store_true',
                    help='use fixed prior, default is learned prior')
parser.add_argument('-noActnorm', '--noActnorm', dest='noActnorm', action='store_true',
                    help='disable actnorm, default uses actnorm')
parser.add_argument('--nonlin', default="elu", type=str, choices=["relu", "elu", "sorting", "softplus"])
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--save_dir', default=None, type=str, help='directory to save results')
parser.add_argument('--vis_port', default=8097, type=int, help="port for visdom")
parser.add_argument('--vis_server', default="localhost", type=str, help="server for visdom")
parser.add_argument('--log_every', default=50, type=int,
                    help='logs every x iters')
parser.add_argument('-log_verbose', '--log_verbose', dest='log_verbose', action='store_true',
                    help='verbose logging: sigmas, max gradient')
parser.add_argument('-deterministic', '--deterministic', dest='deterministic', action='store_true',
                    help='fix random seeds and set cuda deterministic')

# extra fun
parser.add_argument('-cpc', '--cpc', dest='cpc', action='store_true',
                    help='train cpc loss')
parser.add_argument('-sfa', '--sfa', dest='sfa', action='store_true',
                    help='slow feature analysis')
parser.add_argument('--cond_ent_weight', default=1e0, type=float,
                    help='weight of conditional entropy')
parser.add_argument('--marg_ent_weight', default=1e0, type=float,
                    help='weight of marginal entropy')
parser.add_argument('--straightness_weight', default=1e0, type=float,
                    help='weight of straightness')
parser.add_argument('--L1_img_weight', default=1e0, type=float,
                    help='weight of L1_img')
parser.add_argument('--L1_trans_weight', default=1e0, type=float,
                    help='weight of L1_trans')
parser.add_argument('--ce_weight', default=1e0, type=float,
                    help='cross entropy weight')
parser.add_argument('--sfa_batch', default=32, type=int,
                    help='how many videos per sfa batch')
parser.add_argument('--time_depth', default=4, type=int,
                    help='how many frames per sfa video')
parser.add_argument('--step_size', default=5, type=int,
                    help='how many steps between frames')


def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)

try_make_dir('results')

def anaylse_trace_estimation(model, testset, use_cuda, extension):
    # setup range for analysis
    numSamples = np.arange(10)*10 + 1
    numIter = np.arange(10)
    # setup number of datapoints
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    # TODO change
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        # compute trace
        out_bij, p_z_g_y, trace, gt_trace = model(inputs[:, :, :8, :8],
                                                       exact_trace=True)
        trace = [t.cpu().numpy() for t in trace]
        np.save('gtTrace'+extension, gt_trace)
        np.save('estTrace'+extension, trace)
        return
    

def test_spec_norm(model, in_shapes, extension):
    i = 0
    j = 0
    params = [v for v in model.module.state_dict().keys() \
              if "bottleneck" and "weight" in v \
              and not "weight_u" in v \
              and not "weight_orig" in v \
              and not "bn1" in v and not "linear" in v]
    print(len(params))
    print(len(in_shapes))
    svs = [] 
    for param in params:
      if i == 0:
        input_shape = in_shapes[j]
      else:
        input_shape = in_shapes[j]
        input_shape[1] = int(input_shape[1] // 4)

      convKernel = model.module.state_dict()[param].cpu().numpy()
      input_shape = input_shape[2:]
      fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
      t_fft_coeff = np.transpose(fft_coeff)
      U, D, V = np.linalg.svd(t_fft_coeff, compute_uv=True, full_matrices=False)
      Dflat = np.sort(D.flatten())[::-1] 
      print("Layer "+str(j)+" Singular Value "+str(Dflat[0]))
      svs.append(Dflat[0])
      if i == 2:
        i = 0
        j+= 1
      else:
        i+=1
    np.save('singular_values'+extension, svs)
    return


def get_init_batch(dataloader, batch_size):
    """
    gets a batch to use for init
    """
    batches = []
    seen = 0
    for x, y in dataloader:
        batches.append(x)
        seen += x.size(0)
        if seen >= batch_size:
            break
    batch = torch.cat(batches)
    return batch


def main(args, train_log):
    if args.sfa:
        print('Slow Feature Analysis', file=train_log)
    else:
        args.sfa_batch = None
        args.time_depth = None
        args.step_size = None

    if not(args.densityEstimation) and not(args.cpc) and not(args.sfa):
        args.ce_weight = 1e0  # if nothing else, do classification

    if args.deterministic:
        print("MODEL NOT FULLY DETERMINISTIC", file=train_log)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        torch.backends.cudnn.deterministic=True

    dens_est_chain = [
        lambda x: (255. * x) + torch.zeros_like(x).uniform_(0., 1.),
        lambda x: x / 256.,
        lambda x: x - 0.5
    ]
    if args.dataset == 'mnist':
        assert args.densityEstimation, "Currently mnist is only supported for density estimation"
        mnist_transforms = [transforms.Pad(2, 0), transforms.ToTensor(), lambda x: x.repeat((3, 1, 1))]
        transform_train_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
        transform_test_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train_mnist)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=False, transform=transform_test_mnist)
        args.nClasses = 10
        in_shape = (3, 32, 32)
    else:
        if args.dataset == 'svhn':
            train_chain = [transforms.Pad(4, padding_mode="symmetric"),
                           transforms.RandomCrop(32),
                           transforms.ToTensor()]
        else:
            train_chain = [transforms.Pad(4, padding_mode="symmetric"),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()]

        test_chain = [transforms.ToTensor()]
        if args.densityEstimation:
            transform_train = transforms.Compose(train_chain + dens_est_chain)
            transform_test = transforms.Compose(test_chain + dens_est_chain)
        else:
            clf_chain = [transforms.Normalize(mean[args.dataset], std[args.dataset])]
            transform_train = transforms.Compose(train_chain + clf_chain)
            transform_test = transforms.Compose(test_chain + clf_chain)


        if args.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
            args.nClasses = 10
        elif args.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform_test)
            args.nClasses = 100
        elif args.dataset == 'svhn':
            trainset = torchvision.datasets.SVHN(
                root='./data', split='train', download=True, transform=transform_train)
            testset = torchvision.datasets.SVHN(
                root='./data', split='test', download=True, transform=transform_test)
            args.nClasses = 10
        if args.sfa:
            trainset_sfa = UCF101(
                '/gpfs01/bethge/data/ucf101/', time_depth=args.time_depth,
                step_size=args.step_size)
        in_shape = (3, 32, 32)


    # setup logging with visdom
    #viz = visdom.Visdom(port=args.vis_port, server="http://" + args.vis_server)
    #assert viz.check_connection(), "Could not make visdom"

    if args.sfa:
        trainloader_sfa = torch.utils.data.DataLoader(
            trainset_sfa, batch_size=args.sfa_batch, shuffle=True,
            num_workers=2, collate_fn=torch.cat
        )
    if args.deterministic:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                                  shuffle=True, num_workers=2, worker_init_fn=np.random.seed(1234))
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                                 shuffle=False, num_workers=2, worker_init_fn=np.random.seed(1234))
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)

    def get_model(args):
        if args.multiScale:
            model = multiscale_iResNet(in_shape,
                                       args.nBlocks, args.nStrides, args.nChannels,
                                       args.init_ds == 2,
                                       args.inj_pad, args.coeff, args.densityEstimation,
                                       args.nClasses, 
                                       args.numTraceSamples, args.numSeriesTerms,
                                       args.powerIterSpectralNorm,
                                       actnorm=(not args.noActnorm),
                                       learn_prior=(not args.fixedPrior),
                                       nonlin=args.nonlin)
        else:
            model = iResNet(nBlocks=args.nBlocks, nStrides=args.nStrides,
                            nChannels=args.nChannels, nClasses=args.nClasses,
                            init_ds=args.init_ds,
                            inj_pad=args.inj_pad,
                            in_shape=in_shape,
                            coeff=args.coeff,
                            numTraceSamples=args.numTraceSamples,
                            numSeriesTerms=args.numSeriesTerms,
                            n_power_iter = args.powerIterSpectralNorm,
                            density_estimation=args.densityEstimation,
                            actnorm=(not args.noActnorm),
                            learn_prior=(not args.fixedPrior),
                            nonlin=args.nonlin,
                            cpc=args.cpc,
                            sfa=args.sfa,
                            time_depth=args.time_depth,
                            )
        return model

    model = get_model(args)

    # init actnrom parameters
    init_batch = get_init_batch(trainloader, args.init_batch)
    print("initializing actnorm parameters...", file=train_log)
    with torch.no_grad():
        model(init_batch, ignore_logdet=True)
    print("initialized", file=train_log)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
        cudnn.benchmark = True
        in_shapes = model.module.get_in_shapes()
    else:
        in_shapes = model.get_in_shapes()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume), file=train_log)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_objective = checkpoint['objective']
            print('objective: '+str(best_objective), file=train_log)
            model = checkpoint['model']
            if use_cuda:
                model.module.set_num_terms(args.numSeriesTerms)
            else:
                model.set_num_terms(args.numSeriesTerms)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']), file=train_log)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume),
                  file=train_log)

    try_make_dir(args.save_dir)
    if args.analysisTraceEst:
        anaylse_trace_estimation(model, testset, use_cuda, args.extension)
        return

    if args.norm:
        test_spec_norm(model, in_shapes, args.extension) 
        return

    if args.interpolate:
        interpolate(model, testloader, testset, start_epoch, use_cuda, best_objective, args.dataset)
        return

    if args.evaluate:
        test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
        if use_cuda:
            model.module.set_num_terms(args.numSeriesTerms)
        else:
            model.set_num_terms(args.numSeriesTerms)
        model = torch.nn.DataParallel(model.module)
        test(best_objective, args, model, start_epoch, testloader, use_cuda, test_log)
        return

    print('|  Train Epochs: ' + str(args.epochs), file=train_log)
    print('|  Initial Learning Rate: ' + str(args.lr), file=train_log)

    elapsed_time = 0
    test_objective = -np.inf

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)

    with open(os.path.join(args.save_dir, 'params.txt'), 'w') as f:
        f.write(json.dumps(args.__dict__))

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('|  Number of Trainable Parameters: ' + str(params), file=train_log)

    for epoch in range(1, 1+args.epochs):
        start_time = time.time()
        # save checkpoint
        torch.save(model, os.path.join(args.save_dir, 'checkpoint.t7'))
        print('\nEpoch=%s' % epoch, file=train_log)
        if args.sfa:
            train(args, model, optimizer, epoch, trainloader, trainset,
                  use_cuda, train_log, trainloader_sfa, trainset_sfa)
        else:
            train(args, model, optimizer, epoch, trainloader, trainset,
                  use_cuda, train_log)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('\n| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)),
              file=train_log)

    if not args.cpc and not args.sfa:
        print('Testing model', file=train_log)
        test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
        test_objective = test(test_objective, args, model, epoch, testloader, use_cuda, test_log)
        print('* Test results : objective = %.2f%%' % (test_objective),
              file=train_log)
        with open(os.path.join(args.save_dir, 'final.txt'), 'w') as f:
            f.write(str(test_objective))


if __name__ == '__main__':
    args = parser.parse_args()
    with open(os.path.join(args.save_dir, "train_log.txt"), 'w') as f:
        print('Starting Training\n', file=f)
    train_log = open(os.path.join(args.save_dir, "train_log.txt"), 'a', 1)
    sys.stdout = train_log
    try:
        main(args, train_log)
    except Exception:
        traceback.print_exc(file=train_log)
    finally:
        train_log.close()

