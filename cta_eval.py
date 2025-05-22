import argparse
import math

import numpy as np
import torch
import torch.nn as nn
# import wandb

from models.prepare import prepare_model
from utils.dataset import prepare_imagenet_test_data, prepare_cifar10_test_data, prepare_cifar100_test_data, prepare_cifar10_test_data_bybatch,prepare_cifar100_test_data_bybatch, prepare_imagenet_test_data_bybatch,prepare_cifar10_test_data_dirichlet_skew, prepare_cifar100_test_data_dirichlet_skew, prepare_imagenet_test_data_dirichlet_skew
from utils.utils import set_seed, str2bool
from utils.eval import validate, validate_bybatch
from utils.config import set_torch_hub

from utils.cli_utils import AverageMeter, ProgressMeter, accuracy
import matplotlib.pyplot as plt
from utils.datahelper import DataHelper
from algorithm.base import AdaptableModule
import os
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(description='STTA Evaluation')

    # overall experimental settings
    parser.add_argument('--eval_mode', default='continual',
                        choices=['continual','bybatch'], type=str,
                        help='evaluation mode. \n'
                             'continual: Like group, but do not reset model after a limited num of batches. \n'
                             'bybatch: load data batch by batch for extremely memory-constrained devices')
    parser.add_argument('--alg', default='tent', choices=['src', 'bn',  'eata', 'tent', 
                                                         'cotta', 'sar', 'rotta'],
                        type=str, help='algorithms: src - source model;  ')
    parser.add_argument('--no_log', action='store_true', help='disable logging.')
    parser.add_argument('--short', action='store_true', help='activate only for short latency test')
    
    # ONDEVTTA related parmeters
    parser.add_argument('--adaptrate', default=1, type=float, help='Adaptation Rate. The rest will be SKIPPED (Only Forward)')
    parser.add_argument('--adst', default='basic', choices=['basic', 'all', 'high_conf', 'low_entr', 'wdist_custom'],
                        type=str, help='memory add strategy')
    parser.add_argument('--rmst', default='RAND', choices=['RAND', 'FIFO','RS', 'CONF', 'ENTR', 'WASS', 'WASS_OPP'],
                        type=str, help='memory remove strategy')
    parser.add_argument('--memtype', default='pb', choices=['normal', 'pb'],
                        type=str, help='memory type')
    parser.add_argument('--alginf', action='store_true', help='update memory stats at every adaptation')
    parser.add_argument('--mem_size', default=-1, type=int, help='memory size, basically equal to batch size')
    parser.add_argument('--memreset', action='store_true', help='memory reset after adaptation')
    parser.add_argument('--maxage', default=1, type=int, help='maxage for memory')
    parser.add_argument('--confth', default=0.5, type=float, help='threshold for confidence')
    parser.add_argument('--iobmn', action='store_true', help='enable IoBMN')
    parser.add_argument('--iobmn_k', default=10000.0, type=float)
    parser.add_argument('--iobmn_s', default=1.0, type=float)
    parser.add_argument('--use_tb', action='store_true', help='use test batch stats on infer')
    parser.add_argument('--use_mtb', action='store_true', help='use moving test batch stats on infer')
    parser.add_argument('--bn_beta', default=0.1, type=float)
    
    parser.add_argument('--print', default=0, type=int, help='print log every n')

    # path of data, output dir
    parser.add_argument('--data', default='IN', choices=['cifar10', 'IN', 'cifar100'],
                        help='dataset')
    parser.add_argument('--test_corrupt', default='std',
                        help='index of the target sigle corruption. std - standard one used in CTA.')
    parser.add_argument('--model', default='resnet50', type=str)

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2025, type=int, help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda', type=str, help='device to use.')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.00025, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float)

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')

    # batch config for eval
    parser.add_argument('--iters', default=-1, type=int, help='how many iterations for eval. [Default: -1 for all batches]')
    parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default: 16)')
    parser.add_argument('--support_batch', default=None, type=int, help='number of batches for support set (default: 1)')
    parser.add_argument('--merge_batches', default=False, type=str2bool,
                        help='whether to merge several batches of images into one batch. '
                             'Effective w/ group eval. Use this to make a large batch from '
                             'a mixture of data domains.')
    parser.add_argument('--cur_batch', default=1, type=int, help='# of current-domain batches used'
                                                                 'in pair evaluation.')

    # MECTA configuration
    parser.add_argument('--accum_bn', default=False, type=str2bool, help='accumulate BN stats.')
    parser.add_argument('--init_beta', default=None, type=float,
                        help='init beta for accum_bn. Use 1. to avoid using train bn. Default will use the same value as beta.')
    parser.add_argument('--beta', default=0.1, type=float, help='beta for accum_bn.')
    parser.add_argument('--forget_gate', default=False, type=str2bool, help='use forget gate.')
    parser.add_argument('--bn_dist_metric', default='skl', type=str,
                        choices=['kl', 'skl', 'skl2', 'simple', 'mmd'])
    parser.add_argument('--bn_dist_scale', default=1., type=float)

    parser.add_argument('--prune_q', default=0., type=float, help='q is the rate of parameters to remove. If is zero, all parameters will be kept.')
    parser.add_argument('--beta_thre', default=0., type=float, help='minimal threshold for beta to do caching. If is zero, all layers will cache.')


    # for ablation study
    parser.add_argument('--n_layer', type=int, default=None, help='For Tent&EATA, num of BN layers to train, start from the output.')
    parser.add_argument('--layer_grad_chkpt_segment', type=int, default=1, help='Num of segments per ResNet stage for gradient checkpointing.')
    parser.add_argument('--layer_t', type=int, default=0, help='Target layer number to calculate Doamin Centroid.')

    
    args = parser.parse_args()


    # default args
    args.c_margin = args.confth
    if args.mem_size == -1:
        args.mem_size = args.batch_size
    # follow eata settings
    args.fisher_clip_by_norm = 10.
    args.fisher_size = 2000
    if args.data == 'cifar10':
        args.fisher_alpha = 1.
        args.e_margin = math.log(10) * 0.40
        args.d_margin = 0.4
    elif args.data == 'cifar100':
        args.fisher_alpha = 2000.
        args.e_margin = math.log(100) * 0.40
        args.d_margin = 0.05
    elif args.data == 'IN':
        args.fisher_alpha = 2000.
        args.e_margin = math.log(1000) * 0.40
        args.d_margin = 0.05
    else:
        raise NotImplementedError(f'No default EATA param for data: {args.data}')
    return args


def main(args):
    # set random seeds
    if args.seed is not None:
        set_seed(args.seed, True)

    if args.data in ['cifar10', 'cifar100', 'IN', 'cifar10niid','cifar100niid','INniid']:
        corruptions = ['gaussian_noise', 'shot_noise',  'impulse_noise', 'defocus_blur',
                               'glass_blur',     'motion_blur', 'zoom_blur',      'snow',
                               'frost',           'fog',        'brightness',     'contrast',
                               'elastic_transform', 'pixelate', 'jpeg_compression', 'original']
        if args.test_corrupt == 'std':  # for continual eval only
            all_corruptions = corruptions[:-1]
        elif args.test_corrupt == 'long':
            all_corruptions = corruptions[:-1] * 10
        elif args.test_corrupt == 'org':
            all_corruptions = ['original']
        else:
            corruption_indices = [int(x) for x in args.test_corrupt.split(',')]
            all_corruptions = [corruptions[i] for i in corruption_indices]   
    else:
        raise NotImplementedError(f"data: {args.data}")
    print("All corruptions:", all_corruptions)

    print(args)
    
    # Prepare data
    if args.data == 'cifar10':
        prepare_data = prepare_cifar10_test_data
    elif args.data == 'cifar100':
        prepare_data = prepare_cifar100_test_data
    elif args.data == 'IN':
        prepare_data = prepare_imagenet_test_data
    elif args.data == 'cifar10niid':
        prepare_data = prepare_cifar10_test_data_dirichlet_skew
        args.data = 'cifar10'
    elif args.data == 'cifar100niid':
        prepare_data = prepare_cifar100_test_data_dirichlet_skew
        args.data = 'cifar100'
    elif args.data == 'INniid':
        prepare_data = prepare_imagenet_test_data_dirichlet_skew
        args.data = 'IN'
    else:
        raise NotImplementedError(f"data: {args.data}")

    # Prepare models
    subnet = prepare_model(args)
    subnet = subnet.to(args.device)
        
    # Prepare algorithms
    if args.alg == 'src':
        subnet.eval()
        adapt_model = subnet
    elif args.alg == 'bn':
        subnet.train()
        for m in subnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(False)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        else:
            for m in subnet.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(False)
        adapt_model = subnet
    elif args.alg == 'tent':
        from algorithm.tent import Tent
        subnet = Tent.configure_model(subnet)
        params, param_names = Tent.collect_params(subnet)
        optimizer = torch.optim.SGD([{'params': params['affine']}], lr=args.lr,
                                    momentum=args.momentum)
        adapt_model = Tent(subnet, optimizer, args.e_margin, args.maxage, args.c_margin, args.w_min, args.w_max,layer_t=args.layer_t)
    elif args.alg == 'eata':
        from algorithm.eata import EATA, compute_fishers
        subnet = EATA.configure_model(subnet)
        params, param_names = EATA.collect_params(subnet)

        # compute fisher info-matrix
        _, fisher_loader = prepare_data(
            'original', args.level, args.batch_size, workers=args.workers,
            subset_size=args.fisher_size, seed=args.seed + 1)
        fishers = compute_fishers(params['affine'], subnet, fisher_loader, args.device)

        optimizer = torch.optim.SGD(params['affine'], args.lr, momentum=args.momentum)
        adapt_model = EATA(subnet, optimizer, args.maxage, args.c_margin, fishers, args.fisher_alpha,e_margin=args.e_margin, d_margin=args.d_margin,layer_t=args.layer_t)
    elif args.alg == 'cotta':
        from algorithm.cotta import CoTTA
        assert args.n_layer is None, "Not support partial layer."
        subnet = CoTTA.configure_model(subnet)
        params, param_names = CoTTA.collect_params(subnet)
        if args.data == 'cifar10':
            optimizer = torch.optim.Adam(params, lr=args.lr,
                                        betas=(0.9, 0.999), weight_decay=0.)
            cotta_kwargs = dict(mt_alpha=0.999, rst_m=0.01, ap=0.92)
        elif args.data == 'cifar100':
            optimizer = torch.optim.Adam(params, lr=args.lr,
                                        betas=(0.9, 0.999), weight_decay=0.)
            cotta_kwargs = dict(mt_alpha=0.999, rst_m=0.01, ap=0.72)
        elif args.data == 'IN':
            optimizer = torch.optim.SGD(params, lr=args.lr,
                                        momentum=0.9, dampening=0, weight_decay=0., nesterov=True)
            cotta_kwargs = dict()
            from algorithm.cotta import CoTTA_ImageNet as CoTTA
        else:
            raise NotImplementedError(f"data: {args.data}")
        adapt_model = CoTTA(subnet, optimizer,e_margin=args.e_margin, maxage=args.maxage, c_margin=args.c_margin,**cotta_kwargs, device=args.device, layer_t=args.layer_t)
    elif args.alg == 'sar':
        from algorithm.sar import SAR
        from algorithm.sar_sam import SAM
        
        subnet = SAR.configure_model(subnet)
        params, param_names = SAR.collect_params(subnet)

        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9) # sharpness-aware minimization
        adapt_model = SAR(subnet, optimizer, args.e_margin, args.maxage, args.c_margin,layer_t=args.layer_t)
    elif args.alg == 'rotta':
        from algorithm.rotta import RoTTA
        subnet = RoTTA.configure_model(subnet)
        params, param_names = RoTTA.collect_params(subnet)
        optimizer = torch.optim.SGD([{'params': params['affine']}], lr=args.lr,
                                    momentum=args.momentum)

        # for transformation function
        if args.data == 'cifar10' or args.data == 'cifar100':
            input_size = 32
        elif args.data == 'IN':
            input_size = 224

        adapt_model = RoTTA(subnet, optimizer, args.e_margin, args.maxage, args.c_margin, args.w_min, args.w_max, input_size)

    else:
        raise NotImplementedError(f'alg: {args.alg}')
    

    # Start continual adaptation
    if args.eval_mode == 'continual':
        all_adaptrate = []
        all_adaptrate.append(args.adaptrate)
        for i_corrupt, corrupt in enumerate(all_corruptions):
            print('Current corrupt:', corrupt)

            _, val_loader = prepare_data(
                corrupt, args.level, args.batch_size, workers=args.workers)
            
            for i_adaptrate, adaptrate in enumerate(all_adaptrate):
                print('Current adaption rate:', adaptrate)

                acc, max_cache, avg_cache, data_list = validate(args,val_loader, adapt_model, args.device,
                                        stop_at_step=args.iters)
                info = f"[{i_corrupt}] {args.alg}@{corrupt} Acc: {acc:.2f}%"
                print(info)
                
    # For extreme memmory-constrained device
    elif args.eval_mode == 'bybatch':
        all_adaptrate = []
        all_adaptrate.append(args.adaptrate)
        avg_values_list = []
        if args.data == 'cifar10':
            prepare_data = prepare_cifar10_test_data_bybatch
        elif args.data == 'cifar100':
            prepare_data = prepare_cifar100_test_data_bybatch
        elif args.data == 'IN':
            prepare_data = prepare_imagenet_test_data_bybatch
        for i_corrupt, corrupt in enumerate(all_corruptions):
            print('Current corrupt:', corrupt)
            # datahelper util for reading data for batch.
            datahelper = DataHelper(args.data, corrupt, args.level, shuffle=True)
            
            for i_adaptrate, adaptrate in enumerate(all_adaptrate):
                print('Current adaption rate:', adaptrate)

                acc_mt, _, _ = validate_bybatch(args,prepare_data, corrupt, adapt_model, args.device, datahelper,stop_at_step=args.iters)
                
                acc = acc_mt.avg
                avg_values_list.append(acc_mt.avg_values)

                if args.alg in ['eata', 'eta']:
                    print(
                        f"num of reliable samples is {adapt_model.num_samples_update_1}, "
                        f"num of reliable+non-redundant samples is {adapt_model.num_samples_update_2}, "
                        f"num of adapted batch is {adapt_model.num_batch_adapted}, ")
                    adapt_model.num_samples_update_1, adapt_model.num_samples_update_2 = 0, 0
                info = f"[{i_corrupt}] {args.alg}@{corrupt} Acc: {acc:.2f}%"
                print(info)
    else:
        raise NotImplementedError(f"eval mode: {args.eval_mode}")


if __name__ == '__main__':
    set_torch_hub()
    args = get_args()
    main(args)
