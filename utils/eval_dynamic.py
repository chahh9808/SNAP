"""Utils for evaluation"""
import sys
import time
import torch
# import wandb
import numpy as np
from tqdm import tqdm
from algorithm.base import AdaptableModule
from utils.cli_utils import AverageMeter, ProgressMeter, accuracy, MovingAverage
from typing import List
from models.batch_norm import get_last_beta, get_bn_cache_size
from utils.datahelper import DataHelper
from utils.latency_track import TimeTracker

import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
import torch.nn.functional as F




def validate(val_loader, model, device, stop_at_step=-1):
    batch_time = AverageMeter('Time', ':6.3f')
    acc_mt = AverageMeter('Acc', ':6.2f')
    beta_mt = AverageMeter('Beta', ':6.3f')
    beta_std_mt = AverageMeter('Beta std', ':6.3f')
    cache_mt = AverageMeter('Cache', ':6.3f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, acc_mt, beta_mt, beta_std_mt, cache_mt],
        prefix='Test: ')

    with torch.no_grad():
        for i, dl in enumerate(val_loader):
            end = time.time()
            images, target = dl[0], dl[1]
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]
            acc_mt.update(acc1, images.size(0))
            # top5.update(acc5[0], images.size(0))

            # measure elapsed time
            cur_batch_time = time.time() - end
            batch_time.update(cur_batch_time)
            # end = time.time()

            # get AccumBN beta
            # betas = get_last_beta(model)
            # if len(betas) > 0:
            #     beta_mt.update(np.mean(betas))
            #     beta_std_mt.update(np.std(betas))
                # wandb.log({f'beta/layer-{ib}': b for ib, b in enumerate(betas)}, commit=False)
                # wandb.log({f'mean beta': np.mean(betas)}, commit=False)

            # max_forward_cs, backward_cs = get_bn_cache_size(model)
            # if backward_cs is not None and max_forward_cs is not None:
            #     cache_size = max([max_forward_cs, backward_cs])
            #     cache_mt.update(cache_size / 1e6)
                # wandb.log({'cache size (MB)': cache_size / 1e6,}, commit=True)

            if i % 50 == 0:
                progress.display(i)
            # wandb.log({'batch acc': acc1}, commit=True)

            if stop_at_step > 0 and i >= stop_at_step:
                break
    # return acc_mt.avg, cache_mt.max, cache_mt.avg
    return acc_mt.avg, 0, 0

# KDE 점수 계산 함수
def calculate_kde_score(X, bandwidth=1.0):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(X)
    score = kde.score_samples(X)
    return score.mean()

def kl_divergence(p_logits, q_logits):
    p = F.softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    kl_div = F.kl_div(q.log(), p, reduction='batchmean')
    return kl_div

epsilon = 1e-9
def js_divergence(p_logits, q_logits):
    p = F.softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    # 평균 분포 M 계산
    m = 0.5 * (p + q)
    
    # KL divergence 계산
    kl_pm = F.kl_div(p.log(), m, reduction='batchmean') + epsilon
    kl_qm = F.kl_div(q.log(), m, reduction='batchmean') + epsilon
    
    # Jensen-Shannon Divergence
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    return jsd

def validate_bybatch_dynamic(args,prepare_data,corrupt,model, device,datahelper, stop_at_step=-1):
    
    TimeTracker.init_tracker()
    
    acc_mt = AverageMeter('Acc', ':6.2f')
    kde_mt = AverageMeter('KDE', ':6.2f')  
    mem_kde_mt = AverageMeter('MEMKDE', ':6.2f')    
    batch_time = AverageMeter('Total latency(batch)', ':6.4f')
    
    dtload_time = AverageMeter('Data Loading', ':6.4f')
    dtprocess_time = AverageMeter('Data Process(for adapt)', ':6.4f')
    fw_time = AverageMeter('Model Forward', ':6.4f')
    bp_time = AverageMeter('Loss Backprop', ':6.4f')
    optstep_time = AverageMeter('Optim Step', ':6.4f')
    
    batch_num = 10000//args.batch_size
    progress = ProgressMeter(
        batch_num,
        [acc_mt, kde_mt,mem_kde_mt, dtload_time, dtprocess_time, fw_time, bp_time, optstep_time, batch_time],
        prefix='Test: ')
    # datahelper util for reading data for batch.
    # datahelper = DataHelper(args.data, corrupt, args.level, shuffle=True)
    
    prev_acc = 100
    afteradapt = False
    isadapt = True
    cnt = 0
    adaptnum = 0
    js_div = 0
    js_div_aged = 0
    prev_kl = 100
    updatememnum = 0
    # kde_moving = MovingAverage(window_size=10)
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
                    # if isadapt==False:
                    #     isadapt = (idx) % (batch_num // (batch_num*args.adaptrate)) == 0
                    if isadapt:
                        # afteradapt = True
                        adaptnum +=1
                        # print(f"Adapting batch num: {idx}")
                    output = model(images, progress, isadapt, args.memtype,args.adst,args.rmst,args.mem_size, args.memreset, args.upmem)
                
                # if afteradapt:
                # measure accuracy and record loss
                acc1 = accuracy(output, target, topk=(1,))[0]
                acc_mt.update(acc1, images.size(0))
                
                
                # KLD 계산
                if model.prev_memoutput is not None:
                    aged_indices = model.memory.get_aged_indicies()
                    # print(aged_indices)
                    # KL divergence 계산
                    # js_div = js_divergence(model.prev_memoutput, model.memoutput)
                    js_div_aged = js_divergence(model.prev_memoutput[aged_indices], model.memoutput[aged_indices])
                    # if kl_div_aged < 1:
                    #     model.entrth*=0.95
                    # print(js_div_aged)
                    model.prev_memoutput = None
                    # if kl_div < prev_kl:
                    #     cnt+=1
                    # prev_kl=kl_div
                    # progress.display(idx)
                
                # KDE 점수 계산
                current_kde_score = calculate_kde_score(output.detach().cpu().numpy())+12
                # if isadapt and (model.memoutput is not None):
                #     mem_kde_score = calculate_kde_score(model.memoutput.detach().cpu().numpy())+12
                #     mem_kde_mt.update(mem_kde_score*100,1)
                kde_mt.update(current_kde_score*100, 1)
                if idx==0:
                    kde_moving = current_kde_score*100
                
                if isadapt:
                    cnt = 0
                    isadapt = False
                    
                # if js_div_aged < 2.5: 
                #     model.update_mem(model.memoutput)
                #     updatememnum += 1
                #     js_div_aged = 100
                    # isadapt=True
                # else:
                #     isadapt=False   
                # model.memory.prnt_class_dist()
                
                # if acc_mt.avg > acc1: cnt+=1
                
                if kde_moving < current_kde_score*100: cnt+=1
                else: cnt=0
                
                if cnt == 2:
                    isadapt=True
                    
                kde_moving = kde_moving*0.7 + current_kde_score*30
                
                
                # if cnt == 3: 
                #     cnt=0
                #     prev_kl = 100
                #     isadapt=False   
                
                
                
                    # prev_acc = acc1

                # measure elapsed time
                cur_batch_time = time.time() - end
                batch_time.update(cur_batch_time)

            # if idx % 50 == 0:
            #     progress.display(idx)
                # model.memory.print_age_dist()
                # print(adaptnum)
                # model.print_first_bn_layer_stats()
            if stop_at_step > 0 and i >= stop_at_step:
                break
            
            if args.short and idx > 100:
                break
    print(adaptnum)
    print(adaptnum/batch_num)
    print(updatememnum)
    return acc_mt, 0, 0


def group_validate(val_loader, adapt_loaders: List, model, device, adapt_batches=None, n_batch=1,
                   merge_batches=False, stop_at_step=-1,
                   display_interval=50):
    """Validate model using a group of data/batches.

    Args:
        adapt_loaders (list): Loader for adaptation whose acc will not be counted.
        val_loader: Loader for validation whose acc is counted.
        adapt_batches (list): Batch sizes for every adapt_loader. If not provided, equals n_batch-1.
        n_batch (int): Number of batches for adaptation, where we reset model after n batches.
    """
    assert isinstance(adapt_loaders, list)
    batch_time = AverageMeter('Time', ':6.3f')
    acc_mt = AverageMeter('Acc', ':6.2f')
    pair_sup_acc_mt = AverageMeter('Sup-Acc', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    beta_mt = AverageMeter('Beta', ':6.2f')
    beta_std_mt = AverageMeter('Beta std', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, acc_mt, pair_sup_acc_mt, beta_mt, beta_std_mt],
        prefix='Test: ')

    # n_adapt_batch = n_batch - 1  # adaptation-only batches
    if adapt_batches == None:
        assert len(adapt_loaders) == 1
        adapt_batches = [n_batch-1]
    else:
        assert sum(adapt_batches) == n_batch - 1

    with torch.no_grad():
        end = time.time()

        for i, (images, target) in enumerate(tqdm(val_loader)):
            support_acc_mt = AverageMeter('Acc', ':6.2f')
            if merge_batches:
                batch_list = []
            for j, adapt_loader in enumerate(adapt_loaders):
                ada_iter = iter(adapt_loader)
                n_adapt_batch = adapt_batches[j]
                for _ in range(n_adapt_batch):
                    try:
                        ada_imgs, ada_trgs = next(ada_iter)
                    except StopIteration:
                        ada_iter = iter(adapt_loader)
                        ada_imgs, ada_trgs = next(ada_iter)
                    ada_imgs, ada_trgs = ada_imgs.to(device), ada_trgs.to(device)
                    if merge_batches:
                        batch_list.append(ada_imgs)
                    else:
                        ada_output = model(ada_imgs)

                        ada_acc = accuracy(ada_output, ada_trgs, topk=(1,))[0]
                        support_acc_mt.update(ada_acc, images.size(0))

            images = images.to(device)
            target = target.to(device)
            if merge_batches:
                pred_len = len(images)
                images = torch.cat(batch_list + [images], dim=0)
                output = model(images)
                output = output[-pred_len:]
            else:
                output = model(images)
            # measure accuracy and record loss
            acc = accuracy(output, target, topk=(1,))[0]
            acc_mt.update(acc, images.size(0))
            pair_sup_acc_mt.update(support_acc_mt.avg, 1)

            # measure elapsed time
            cur_batch_time = time.time() - end
            batch_time.update(cur_batch_time)
            end = time.time()

            # get AccumBN beta
            betas = get_last_beta(model)
            beta_mt.update(np.mean(betas))
            beta_std_mt.update(np.std(betas))

            if i % display_interval == 0:
                progress.display(i, print_fh=lambda s: tqdm.write(s, file=sys.stdout))
            # wandb.log({'batch acc': acc}, commit=True)

            if isinstance(model, AdaptableModule):
                # print(f"Reset at batch {i}")
                model.reset_all()
            else:
                model.reset()

            if stop_at_step > 0 and i >= stop_at_step:
                break
    return acc_mt.avg  # acc_mt.step_avg, acc_mt.step_std, acc_mt.update_cnt
