"""Utils for evaluation"""
import sys
import time
import torch
# import wandb
import numpy as np
from tqdm import tqdm
from algorithm.base import AdaptableModule
from utils.cli_utils import AverageMeter, ProgressMeter, accuracy
from typing import List
from models.batch_norm import get_last_beta, get_bn_cache_size
from utils.datahelper import DataHelper
from utils.latency_track import TimeTracker

import torch.nn.functional as F

from utils.plot import plot_label_flip, plot_avg_accuracy, plot_wass_correct, plot_conf_correct

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

@torch.jit.script
def get_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Get confidence from logits."""
    probabilities = logits.softmax(1)
    confidence, _ = torch.max(probabilities, dim=1)
    return confidence    

def validate(args, val_loader, model, source_subnet, device, stop_at_step=-1):
    TimeTracker.init_tracker()
    
    acc_mt = AverageMeter('Acc', ':6.2f')
    batch_time = AverageMeter('Total latency(batch)', ':6.4f')
    
    dtload_time = AverageMeter('Data Loading', ':6.4f')
    dtprocess_time = AverageMeter('Data Process(for adapt)', ':6.4f')
    fw_time = AverageMeter('Model Forward', ':6.4f')
    bp_time = AverageMeter('Loss Backprop', ':6.4f')
    optstep_time = AverageMeter('Optim Step', ':6.4f')

    if args.prelim is True:
        data_list = []
        acc_list = []
        wass_correctness_list = []
        conf_correctness_list = []
        if args.label_flip is True:
            flip_list = []
    else:
        data_list = None
    
    batch_num = len(val_loader)
    progress = ProgressMeter(
        batch_num,
        [acc_mt, dtload_time, dtprocess_time, fw_time, bp_time, optstep_time, batch_time],
        prefix='Test: ')

    with torch.no_grad():
        for i, dl in enumerate(val_loader):
            end = time.time()
            images, target = dl[0], dl[1]
            images = images.to(device)
            target = target.to(device)
            
            if args.alg == 'src' or args.alg =='bn':
                output = model(images)
                TimeTracker.track(fw_time)
            else:
                isadapt = i % (batch_num // (batch_num*args.adaptrate)) == 0
                output = model(images, progress, isadapt, args.memtype,args.adst,args.rmst,args.mem_size, args.memreset, args.alginf)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]
            acc_mt.update(acc1, images.size(0))
            # measure elapsed time
            cur_batch_time = time.time() - end
            batch_time.update(cur_batch_time)
            # end = time.time()

            if args.print > 0:
                if i % args.print == 0:
                    progress.display(i)
            # wandb.log({'batch acc': acc1}, commit=True)

            if stop_at_step > 0 and i >= stop_at_step:
                break

    return acc_mt.avg, 0, 0, data_list

def validate_bybatch(args,prepare_data,corrupt,model, device,datahelper, stop_at_step=-1):
    
    TimeTracker.init_tracker()
    
    acc_mt = AverageMeter('Acc', ':6.2f')
    batch_time = AverageMeter('Total latency(batch)', ':6.4f')
    
    dtload_time = AverageMeter('Data Loading', ':6.4f')
    dtprocess_time = AverageMeter('Data Process(for adapt)', ':6.4f')
    fw_time = AverageMeter('Model Forward', ':6.4f')
    bp_time = AverageMeter('Loss Backprop', ':6.4f')
    optstep_time = AverageMeter('Optim Step', ':6.4f')
    
    batch_num = 10000//args.batch_size
    progress = ProgressMeter(
        batch_num,
        [acc_mt, dtload_time, dtprocess_time, fw_time, bp_time, optstep_time, batch_time],
        prefix='Test: ')
    # datahelper util for reading data for batch.
    # datahelper = DataHelper(args.data, corrupt, args.level, shuffle=True)

    with torch.no_grad():
        for idx in range(batch_num):
            TimeTracker.set_timestamp()
            end = time.time() #for batch time
            _, val_loader = prepare_data(
                corrupt, args.level, args.batch_size, workers=args.workers, idx=idx, datahelper=datahelper)
            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                images = images.to(device);
                target = target.to(device)
                
                TimeTracker.track(dtload_time)
                
                # if args.alg == 'dua' and idx%(batch_num//10)==0:
                #     model.adapt(images,args.batch_size)
                #     TimeTracker.track(dtprocess_time)
                                
                if args.alg == 'src' or args.alg =='bn':
                    output = model(images)
                    TimeTracker.track(fw_time)
                else:
                    isadapt = idx % (batch_num // (batch_num*args.adaptrate)) == 0
                    # isadapt = True
                    # if isadapt:
                    #     print(f"Adapting batch num: {idx}")
                    output = model(images, progress, isadapt, args.memtype,args.adst,args.rmst,args.mem_size, args.memreset, args.alginf)
                
                # measure accuracy and record loss
                acc1 = accuracy(output, target, topk=(1,))[0]
                acc_mt.update(acc1, images.size(0))

                # measure elapsed time
                cur_batch_time = time.time() - end
                batch_time.update(cur_batch_time)

            if args.print > 0:
                if idx % args.print == 0:
                    progress.display(idx)
                    # model.print_first_ln_layer_stats()
            if stop_at_step > 0 and i >= stop_at_step:
                break
            
            if args.short and idx > 100:
                break
        
    return acc_mt, 0, 0