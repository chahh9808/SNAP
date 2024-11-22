"""CoTTA
Source: https://github.com/qinenergy/cotta/blob/main/cifar/cotta.py
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
from algorithm import cotta_transforms
from utils.latency_track import TimeTracker
from models.batch_norm import MectaNorm2d
from utils.cpu_mem_track import MemTracker

import math
from utils.memory import NMemory, PBMemory
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

from utils.iobmn import SparseAdaptationAwareBatchNorm2d, SparseAdaptationAwareLayerNorm
from utils.bn_utils import bn_iobmn_get_bn_stats, bn_iobmn_get_bn_stats_N, bn_retrieve_bn_stats, bn_retrieve_bn_stats_N, bn_recalc_bn_stats, bn_check_bn_divergence

N = 8
eata_thr_c10 = math.log(10) * 0.40
eata_thr_c100 = math.log(100) * 0.40
def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False, img_size=32):
    img_shape = (img_size, img_size, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        cotta_transforms.Clip(0.0, 1.0), 
        cotta_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        cotta_transforms.GaussianNoise(0, gaussian_std),
        cotta_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, e_margin, maxage, c_margin, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, device=None,layer_t=0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
            
        self.model_ema_init = deepcopy(self.model_ema)
        self.model_anchor_init = deepcopy(self.model_anchor)
        self.transform = get_tta_transforms()    
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.device = device
        
        # Memory for effective adapting
        self.mem = None
        self.memory = None
        self.memorytwo = None
        self.memoutput = None
        self.prev_memoutput = None
        self.entrth_init = e_margin
        self.entrth = e_margin
        self.confth = c_margin
        self.maxage = maxage
        
        # domain centroid (mu,sigma2)
        self.mu_centr = None
        self.sigma2_centr = None
        
        self.layer_t = layer_t
        
        # for prelim experiment
        self.time = 0
        
        self.step = 0
        self.memory_stats = [] # adaptation sample analysis
        self.sample_number_stats = [] # threshold-abiding sample analysis
        self.bn_analysis = [] # bn stats (wasserstein distance) analysis
        self.wass_dist = [] # wass dist btw prev bn stats
        self.cnt = 0
        
        self.hidden_features_test = []
        self.hidden_features_test2 = []
        self.hidden_features_mem = []

        # for WDIST calc modification
        self.norm_beta = 0.01
        self.wasserstein_means = None
        self.wasserstein_vars = None
        
        # self.global_entropys = []
        # self.global_confidences = []

    def forward(self, x, progress, isadapt, memtype, adst, rmst, mem_size, memreset,alginf=False):
        if self.episodic:
            self.reset()
        if isadapt:
            if self.memory is None:
                self.model.train()
                self.model_anchor.train()
                self.model_ema.train()
                self.switch_bn(True)
                outputs = self.forward_and_adapt(x,self.optimizer,progress)
                return outputs
            
            # MemTracker.track('Before inference') 
            self.model.eval()
            self.model_anchor.eval()
            self.model_ema.eval()
            self.switch_bn(False)
            with torch.no_grad():
                if alginf:
                    outputs, outputs_ori = self.forward_only(x)
                else:
                    outputs = self.model(x)
                    outputs_ori = outputs
                    
            wdists_test, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt,self.layer_t)
            self.check_bn_divergence(memtype,mu, sigma2)        
            
                    
            # TimeTracker.track(progress.get_meter('fw_time'))   
            # MemTracker.track('Do inference')   
            sample_number = self.add_mem(x,memtype,adst,rmst,mem_size,isadapt,outputs_ori, wdists_test,stats_list)
            
            self.model.train()
            self.model_anchor.train()
            self.model_ema.train()
            self.switch_bn(True)
            
            for _ in range(self.steps):
                _ = self.adapt(self.optimizer,progress,self.mem,rmst)
                
            if adst=='high_low':  
                _ = self.model(torch.stack(self.memory.get_memory()))
            if memreset:
                self.mem = None
        else:
            self.model.eval()
            self.model_anchor.eval()
            self.model_ema.eval()
            self.switch_bn(False)
            with torch.no_grad():
                if alginf:
                    outputs, outputs_ori = self.forward_only(x)
                else:
                    outputs = self.model(x)
                    outputs_ori = outputs
            if adst == 'basic':
                return outputs
            
            wdists_test, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt,self.layer_t)
            self.check_bn_divergence(memtype,mu, sigma2)
            # TimeTracker.track(progress.get_meter('fw_time'))
            
            sample_number = self.add_mem(x, memtype, adst, rmst, mem_size, isadapt, outputs_ori, wdists_test,stats_list)
        self.sample_number_stats.append((self.time, sample_number))
        return outputs
    
    def reset_all(self):
        self.reset()
        self.reset_steps(1)
        self.reset_bn()
        self.mem = None
        self.memory = None
        self.memoutput = None
        self.prev_memoutput = None
        self.entrth = self.entrth_init
        self.global_entropys = []
        self.global_confidences = []
    
    def add_mem(self,x, memtype, adst, rmst, mem_size, isadapt, logits, bn_w_dist,stats_list):
        if adst == 'basic':
            self.mem = x
            return 0
        entropys = softmax_entropy(logits)
        confidences = get_confidence(logits)
        
        if self.memory is None or self.memorytwo is None:
            if memtype == 'normal':
                self.memory = NMemory(capacity=mem_size,num_class=logits.shape[-1],max_age_threshold=self.maxage)
                self.memorytwo = NMemory(capacity=mem_size,num_class=logits.shape[-1],max_age_threshold=self.maxage)
            elif memtype == 'pb':
                self.memory = PBMemory(capacity=mem_size,num_class=logits.shape[-1],max_age_threshold=self.maxage)
                self.memorytwo = PBMemory(capacity=mem_size,num_class=logits.shape[-1],max_age_threshold=self.maxage)
            else:
                assert('NONE PREDEFINED MEMORY TYPE')
            sample_number = len(x)
            for i, x_ins in enumerate(x):
                logit = logits[i]
                pseudo_cls = logit.max(dim=0)[1]
                self.memory.add_instance([x_ins, entropys[i], confidences[i], 0, bn_w_dist[i], stats_list[i], pseudo_cls], rmst)
                
        else: 
            if adst == 'all':
                sample_number = len(x)
                for i, x_ins in enumerate(x):
                    logit = logits[i]
                    pseudo_cls = logit.max(dim=0)[1]
                    self.memory.add_instance([x_ins, entropys[i], confidences[i],0, bn_w_dist[i], stats_list[i], pseudo_cls],rmst)
            elif adst == 'high_conf':
                # high_w_distance_ids = [i for i, dist in enumerate(tr_w_dist) if dist > tr_w_dist_thr]
                hiconf_ids = torch.where((confidences > self.confth) & (confidences < 1))[0].tolist()
                # ids = list(set(high_w_distance_ids) & set(hiconf_ids))
                sample_number = len(hiconf_ids)
                for idx in hiconf_ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, bn_w_dist[idx],stats_list[idx], pseudo_cls],rmst)
            elif adst == 'low_entr':
                lowentro_ids = torch.where(entropys < self.entrth)[0].tolist()
                sample_number = len(lowentro_ids)
                for idx in lowentro_ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx],0, bn_w_dist[idx], stats_list[idx],pseudo_cls],rmst)
            elif adst == 'wdist_custom': # custom min/max threshold from input
                w_min = self.w_min
                w_max = self.w_max
                ids = [i for i, dist in enumerate(bn_w_dist) if dist > w_min and dist < w_max]
                sample_number = len(ids)
                for idx in ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, bn_w_dist[idx], stats_list[idx], pseudo_cls], rmst)

        if isadapt:
            self.mem = torch.stack(self.memory.get_memory())
                
        return sample_number
            
    def update_mem(self,logits):
        if self.mem is not None:
            self.memory.reset_memory()
            entropys = softmax_entropy(logits)
            confidences = get_confidence(logits) 
            for i, x_ins in enumerate(self.mem):
                logit = logits[i]
                pseudo_cls = logit.max(dim=0)[1]
                self.memory.add_instance([x_ins,entropys[i],confidences[i],pseudo_cls])
            if self.memory.get_occupancy() != self.memory.capacity:
                assert("Memory length doesnt match after update")
                    
        
    def reset_bn(self):
        for m in self.model.modules():
            if isinstance(m, MectaNorm2d):
                m.reset()

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        # self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
        #     copy_model_and_optimizer(self.model, self.optimizer)
        self.model_ema = deepcopy(self.model_ema_init)
        self.model_anchor = deepcopy(self.model_anchor_init)
    
    def switch_bn(self,adapt=True):
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                # if filter is not None and not filter(nm):
                #     continue
                m.requires_grad_(adapt)
                m.momentum = 1
                m.track_running_stats = adapt # update moving bn stat
    # def switch_bn(self,adapt=True,model=None):
    #     for nm, m in model.named_modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             if filter is not None and not filter(nm):
    #                 continue
    #             m.requires_grad_(adapt)
    #             m.momentum = 1
    #             m.track_running_stats = adapt # update moving bn stat
                
    def print_first_bn_layer_stats(self):

        bn_layer = None
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                bn_layer = layer
                break

        if bn_layer is None:
            print("모델에 BN 레이어가 없습니다.")
            return

        print("첫 번째 BN 레이어의 통계:")
        print("  배치 평균:", bn_layer.running_mean)
        print("  배치 분산:", bn_layer.running_var)
        print("  gamma:", bn_layer.weight)
        print("  beta:", bn_layer.bias)
        
    def forward_only(self, x): 
        outputs = self.model(x)
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        outputs_emas = []
        for i in range(N):
            outputs_  = self.model_ema(self.transform(x)).detach()
            outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if anchor_prob.mean(0)<self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        return outputs_ema, outputs
        
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, optimizer, progress):        
        outputs = self.model(x)
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        outputs_emas = []
        for i in range(N):
            outputs_  = self.model_ema(self.transform(x)).detach()
            outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if anchor_prob.mean(0)<self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Student update
        loss = (softmax_entropy_cotta(outputs, outputs_ema)).mean(0) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        if self.device == 'cuda':
                            mask = (torch.rand(p.shape)<self.rst).float().cuda()
                        else:
                            mask = (torch.rand(p.shape)<self.rst).float()
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def adapt(self, optimizer, progress, mem,rmst,alginf=False):        
        ######################### For Adapt w memory #######################
        if mem is not None and self.memory is not None:
            outputs_withmem_list = []
            outputs_ema_list = []
            mem_dataset = TensorDataset(mem)  # mem이 torch.Tensor 타입이라고 가정합니다.
            mem_loader = DataLoader(mem_dataset, batch_size=self.memory.get_occupancy(), shuffle=True)
            for mem_batch in mem_loader:
                mem_batch = mem_batch[0] 
                outputs_withmem = self.model(mem_batch)
                outputs_withmem_list.append(outputs_withmem)
            
                # outputs = model(mem)
                # self.prev_memoutput = self.memoutput
                # self.memoutput = outputs
                # if rmst != 'RAND' and upmem:
                #     self.update_mem(outputs)
                
                # Teacher Prediction
                anchor_prob = torch.nn.functional.softmax(self.model_anchor(mem_batch), dim=1).max(1)[0]
                standard_ema = self.model_ema(mem_batch)
                # Augmentation-averaged Prediction
                outputs_emas = []
                for i in range(N):
                    outputs_  = self.model_ema(self.transform(mem_batch)).detach()
                    outputs_emas.append(outputs_)
                # Threshold choice discussed in supplementary
                if anchor_prob.mean(0)<self.ap:
                    outputs_ema = torch.stack(outputs_emas).mean(0)
                else:
                    outputs_ema = standard_ema
                outputs_ema_list.append(outputs_ema)
        ############################################################
            outputs_ema = torch.cat(outputs_ema_list, dim=0)
            # 리스트에 있는 모든 결과를 하나의 텐서로 결합
            outputs = torch.cat(outputs_withmem_list, dim=0)
        else:
            assert('error on adapt without memory')
        # Student update
        loss = (softmax_entropy_cotta(outputs, outputs_ema)).mean(0) 
        loss.backward()
        
        # TimeTracker.track(progress.get_meter('bp_time'))
        optimizer.step()
        optimizer.zero_grad()
        
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        if self.device == 'cuda':
                            mask = (torch.rand(p.shape)<self.rst).float().cuda()
                        else:
                            mask = (torch.rand(p.shape)<self.rst).float() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
                            
        TimeTracker.track(progress.get_meter('bp_time'))
        return outputs_ema
    
    def reset_steps(self, new_steps):
        self.steps = new_steps

    @staticmethod
    def collect_params(model):
        """Collect all trainable parameters.

        Walk the model's modules and collect all parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    @staticmethod
    def configure_model(model):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what we update
        model.requires_grad_(False)
        # enable all trainable
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = True
                m.momentum = 1
                # m.running_mean = None
                # m.running_var = None
            else:
                m.requires_grad_(True)
        return model
    
    def plot_entr_conf(self):

        plt.figure(figsize=(10, 10))

        plt.subplot(2, 1, 1)
        plt.plot(self.global_entropys, label='Entropy')
        plt.title('Entropy')
        plt.xlabel('Data Index')
        plt.ylabel('Entropy Value')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.global_confidences, label='Confidence')
        plt.title('Confidence')
        plt.xlabel('Data Index')
        plt.ylabel('Confidence Value')
        plt.legend()

        plt.tight_layout()
        return plt
    
    def iobmn_get_bn_stats(self, compare_with_test=False,layer_t=0):
        return bn_iobmn_get_bn_stats(self, compare_with_test,layer_t)
            
    def iobmn_get_bn_stats_N(self, N, compare_with_test=False):
        return bn_iobmn_get_bn_stats_N(self, N, compare_with_test)
            
    def retrieve_bn_stats(self, outputs, isadapt,layer_t=0):
        return bn_retrieve_bn_stats(self, outputs, isadapt,layer_t)

    def retrieve_bn_stats_N(self, outputs, N, isadapt):
        return bn_retrieve_bn_stats_N(self, outputs, N, isadapt)

    def recalc_bn_stats(self, memtype,compare_with_test=False):
        return bn_recalc_bn_stats(self,memtype, compare_with_test)

    def check_bn_divergence(self, memtype,mu, sigma2):
        return bn_check_bn_divergence(self,memtype, mu, sigma2)
    
    # gets first layer BN stats
    # def iobmn_get_bn_stats(self, compare_with_test=False):
    #     for module in self.model_ema.modules():
    #         if isinstance(module, SparseAdaptationAwareBatchNorm2d) or isinstance(module, SparseAdaptationAwareLayerNorm):
    #             if compare_with_test:
    #                 return module.mu_batch, module.sigma2_batch, module.mu_prev, module.sigma2_prev
    #             else:
    #                 return module.mu_batch, module.sigma2_batch, module._bn.training_mean, module._bn.training_varå
        
    #     print("ERROR: SparseAdaptationAwareNorm not found")
    #     return None, None, None, None
    
    # # gets first layer BN stats
    # def iobmn_get_bn_stats_test(self, compare_with_test=False):
    #     for module in self.model_ema.modules():
    #         if isinstance(module, SparseAdaptationAwareBatchNorm2d) or isinstance(module, SparseAdaptationAwareLayerNorm):
    #             if compare_with_test:
    #                 return module.mu_batch, module.sigma2_batch, module.mu_prev, module.sigma2_prev, module.mu_test, module.sigma2_test
    #             else:
    #                 return module.mu_batch, module.sigma2_batch, module._bn.training_mean, module._bn.training_var, module.mu_cur, module.sigma2_cur
        
    #     print("ERROR: SparseAdaptationAwareNorm not found")
    #     return None, None, None, None, None, None
    
    # def iobmn_update_prev(self):
    #     for module in self.model_ema.modules():
    #         if isinstance(module, SparseAdaptationAwareBatchNorm2d) or isinstance(module, SparseAdaptationAwareLayerNorm):
    #             module.update_prev()
    #             return None
    
    # # returns first N BN layer stats
    # def iobmn_get_bn_stats_N(self, N, compare_with_test=False):
    #     # N=None means iterate through all BN layers
    #     stats = []
    #     count = 0
    #     for module in self.model.modules():
    #         if isinstance(module, SparseAdaptationAwareBatchNorm2d) or isinstance(module, SparseAdaptationAwareLayerNorm):
    #             if compare_with_test:
    #                 stats.append((module.mu_batch, module.sigma2_batch, module.mu_test, module.sigma2_test))
    #             else:
    #                 stats.append((module.mu_batch, module.sigma2_batch, module._bn.training_mean, module._bn.training_var))
    #             count += 1
    #             if N is not None and count == N:
    #                 break
        
    #     if N is not None and len(stats) != N:
    #         print(f"ERROR: fewer than {N} SparseAdaptationAwareBatchNorm2d were found")
    #         return None
        
    #     return stats
    # # calculate WDIST using first layer BN stats and return list of WDIST for each image
    # def retrieve_bn_stats(self, outputs, compare_with_test=False):
    #     mu_batch, sigma2_batch, mu, sigma2 = self.iobmn_get_bn_stats(compare_with_test)
    #     entropies = softmax_entropy(outputs)

    #     if not compare_with_test:
    #         mu = torch.tensor(mu).to('cuda')
    #         sigma2 = torch.tensor(sigma2).to('cuda')
            
    #     stats_list = []
    
    #     # wasserstein distance
    #     wasserstein_distances = []

    #     # iterate over each image
    #     for i in range(outputs.size(0)):
    #         # wasserstein distance
    #         w_dist = wasserstein_distance(mu_batch[i], sigma2_batch[i], mu, sigma2)

    #         # store entropy and distances as a tuple, and append to global list
    #         per_image_stat = (entropies[i].item(), w_dist)
    #         self.bn_analysis.append(per_image_stat)

    #         wasserstein_distances.append(w_dist)
            
    #         stats_list.append([mu_batch[i],sigma2_batch[i]])

    #     return wasserstein_distances, stats_list, mu, sigma2
    
    # def recalc_bn_stats(self, memtype, compare_with_test=False):
    #     _, _, mu, sigma2 = self.iobmn_get_bn_stats(compare_with_test)
    #     tot_idx = 0
    #     if self.memory is not None:
    #         if memtype == 'pb':
    #             for data_per_cls in self.memory.data:
    #                 for idx in range(len(data_per_cls[3])):
    #                     # print(data_per_cls[4][idx])
    #                     # print(data_per_cls[5][idx])
    #                     w_dist = wasserstein_distance(data_per_cls[5][idx][0], data_per_cls[5][idx][1], mu, sigma2)
    #                     data_per_cls[4][idx] = w_dist
    #                     tot_idx=tot_idx+1
    #                     # print(data_per_cls[4][idx])
    #                     # print('-----------------------')
    #         else:
    #             for idx in range(len(self.memory.data[3])):
    #                 w_dist = wasserstein_distance(self.memory.data[5][idx][0], self.memory.data[5][idx][1], mu, sigma2)
    #                 self.memory.data[4][idx] = w_dist
    #                 tot_idx=tot_idx+1
    #         # print(tot_idx)
    #         # assert(0)
    #     self.cnt = self.cnt+1
    #     return tot_idx

    # # calculate WDIST using first N layer BN stats, normalize each layer WDIST using running average of WDIST stats so that each layer gets fair weight,
    # # then sums all layers to get one WDIST per image
    # # return list of WDIST for each image
    # def retrieve_bn_stats_N(self, outputs, N, compare_with_test=False):
    #     stats = self.iobmn_get_bn_stats_N(N, compare_with_test)
    #     entropies = softmax_entropy(outputs)

    #     wasserstein_distances = [[] for _ in range(N)]

    #     # iterate over each image
    #     for i in range(outputs.size(0)):
    #         # iterate over each BN layer
    #         for layer_idx, stat in enumerate(stats):
    #             mu_batch, sigma2_batch, mu, sigma2 = stat

    #             if not compare_with_test:
    #                 mu = torch.tensor(mu).to('cuda')
    #                 sigma2 = torch.tensor(sigma2).to('cuda')

    #             # wasserstein distance per layer
    #             w_dist = wasserstein_distance(mu, sigma2, mu_batch[i], sigma2_batch[i])
    #             wasserstein_distances[layer_idx].append(w_dist)

    #     wasserstein_means = [torch.mean(torch.tensor(distances)) for distances in wasserstein_distances] # mean of WDIST for each layer 
    #     wasserstein_vars = [torch.var(torch.tensor(distances)) for distances in wasserstein_distances] # variance of WDIST for each layer

    #     # update running estimate of wasserstein_means and wasserstein_vars
    #     if self.wasserstein_means is None and self.wasserstein_vars is None:
    #         self.wasserstein_means = wasserstein_means
    #         self.wasserstein_vars = wasserstein_vars
    #     else:
    #         # use momentum update
    #         self.wasserstein_means = [self.norm_beta * batch_mean + (1 - self.norm_beta) * running_mean for batch_mean, running_mean in zip(wasserstein_means, self.wasserstein_means)]
    #         self.wasserstein_vars = [self.norm_beta * batch_var + (1 - self.norm_beta) * running_var for batch_var, running_var in zip(wasserstein_vars, self.wasserstein_vars)]

    #     normalized_wasserstein_distances = []

    #     # normalize the per layer WDISTs and sum them to get per image WDIST
    #     for i in range(outputs.size(0)):
    #         w_dist_sum = 0
            
    #         for layer_idx in range(N):
    #             w_dist = wasserstein_distances[layer_idx][i]

    #             mean = self.wasserstein_means[layer_idx]
    #             var = self.wasserstein_vars[layer_idx]
    #             w_dist_normalized = torch.abs((w_dist - mean) / torch.sqrt(var + 1e-8))
                
    #             w_dist_sum += w_dist_normalized.item()

    #         # store entropy and WDIST as a tuple, and append to global list
    #         per_image_stat = (entropies[i].item(), w_dist_sum)
    #         self.bn_analysis.append(per_image_stat)

    #         # append stats to list to be returned
    #         normalized_wasserstein_distances.append(w_dist_sum)

    #     return normalized_wasserstein_distances

    


class CoTTA_ImageNet(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.

    ImageNet version
    """

    def __init__(self, model, optimizer, e_margin, maxage, c_margin, steps=1, episodic=False, device=None,layer_t=0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms(img_size=224)
        
        self.device = device
        
        # Memory for effective adapting
        self.mem = None
        self.memory = None
        self.memorytwo = None
        self.memoutput = None
        self.prev_memoutput = None
        self.entrth_init = e_margin
        self.entrth = e_margin
        self.confth = c_margin
        self.maxage = maxage
        
        # for prelim experiment
        self.time = 0
        
        self.step = 0
        self.memory_stats = [] # adaptation sample analysis
        self.sample_number_stats = [] # threshold-abiding sample analysis
        self.bn_analysis = [] # bn stats (wasserstein distance) analysis
        self.wass_dist = [] # wass dist btw prev bn stats
        self.cnt = 0
        
        self.hidden_features_test = []
        self.hidden_features_test2 = []
        self.hidden_features_mem = []
        
        # domain centroid (mu,sigma2)
        self.mu_centr = None
        self.sigma2_centr = None
        
        self.layer_t = layer_t

        # for WDIST calc modification
        self.norm_beta = 0.01
        self.wasserstein_means = None
        self.wasserstein_vars = None

    def forward(self, x, progress, isadapt, memtype, adst, rmst, mem_size, memreset,alginf=False):
        if self.episodic:
            self.reset()

        if isadapt:
            if self.memory is None:
                self.model.train()
                self.model_anchor.train()
                self.model_ema.train()
                self.switch_bn(True)
                outputs = self.forward_and_adapt(x,self.optimizer,progress)
                return outputs
            
            MemTracker.track('Before inference') 
            self.model.eval()
            self.model_anchor.eval()
            self.model_ema.eval()
            self.switch_bn(False)
            with torch.no_grad():
                if alginf:
                    outputs, outputs_ori = self.forward_only(x)
                else:
                    outputs = self.model(x)
            # TimeTracker.track(progress.get_meter('fw_time'))   
            # MemTracker.track('Do inference')   
            
            wdists_test, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt,self.layer_t)
            self.check_bn_divergence(memtype,mu, sigma2)
            
            sample_number = self.add_mem(x,memtype,adst,rmst,mem_size,isadapt,outputs_ori,wdists_test,stats_list)
            
            self.model.train()
            self.model_anchor.train()
            self.model_ema.train()
            self.switch_bn(True)
            for _ in range(self.steps):
                _ = self.adapt(self.optimizer,progress,self.mem,rmst)
            if adst=='high_low':  
                _ = self.model(torch.stack(self.memory.get_memory()))
            if memreset:
                self.mem = None
        else:
            self.model.eval()
            self.model_anchor.eval()
            self.model_ema.eval()
            self.switch_bn(False)
            with torch.no_grad():
                if alginf:
                    outputs, outputs_ori = self.forward_only(x)
                else:
                    outputs = self.model(x)
                    outputs_ori = outputs
            if adst == 'basic':
                return outputs
            # TimeTracker.track(progress.get_meter('fw_time'))
            wdists_test, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt,self.layer_t)
            self.check_bn_divergence(memtype,mu, sigma2)
            
            
            sample_number = self.add_mem(x, memtype, adst, rmst, mem_size, isadapt, outputs_ori, wdists_test,stats_list)
            
        return outputs
    
    def reset_all(self):
        self.reset()
        self.reset_steps(1)
        self.reset_bn()
        self.mem = None
        self.memory = None
        self.memorytwo = None
        self.memoutput = None
        self.prev_memoutput = None
        self.entrth = self.entrth_init
        # self.global_entropys = []
        # self.global_confidences = []
    
    def add_mem(self,x,memtype,adst,rmst,mem_size,isadapt,logits,bn_w_dist,stats_list):
        if adst == 'basic':
            self.mem = x
            return 0
        
        entropys = softmax_entropy(logits)
        confidences = get_confidence(logits)
        if self.memory is None or self.memorytwo is None:
            if memtype == 'normal':
                self.memory = NMemory(capacity=mem_size,num_class=logits.shape[-1],max_age_threshold=self.maxage)
                self.memorytwo = NMemory(capacity=mem_size,num_class=logits.shape[-1],max_age_threshold=self.maxage)
            elif memtype == 'pb':
                self.memory = PBMemory(capacity=mem_size,num_class=logits.shape[-1],max_age_threshold=self.maxage)
                self.memorytwo = PBMemory(capacity=mem_size,num_class=logits.shape[-1],max_age_threshold=self.maxage)
            else:
                assert('NONE PREDEFINED MEMORY TYPE')
            for i, x_ins in enumerate(x):
                logit = logits[i]
                pseudo_cls = logit.max(dim=0)[1]
                self.memory.add_instance([x_ins, entropys[i], confidences[i], 0, bn_w_dist[i], stats_list[i], pseudo_cls], rmst)
                
        else:        
            if adst == 'all':
                sample_number = len(x)
                for i, x_ins in enumerate(x):
                    logit = logits[i]
                    pseudo_cls = logit.max(dim=0)[1]
                    self.memory.add_instance([x_ins, entropys[i], confidences[i],0, bn_w_dist[i], stats_list[i], pseudo_cls],rmst)
            elif adst == 'high_conf':
                hiconf_ids = torch.where((confidences > self.confth) & (confidences < 1))[0].tolist()
                sample_number = len(hiconf_ids)
                for idx in hiconf_ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, bn_w_dist[idx],stats_list[idx], pseudo_cls],rmst)
            elif adst == 'low_entr':
                lowentro_ids = torch.where(entropys < self.entrth)[0].tolist()
                sample_number = len(lowentro_ids)
                for idx in lowentro_ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx],0, bn_w_dist[idx], stats_list[idx],pseudo_cls],rmst)
            elif adst == 'wdist_custom': # custom min/max threshold from input
                w_min = self.w_min
                w_max = self.w_max
                ids = [i for i, dist in enumerate(bn_w_dist) if dist > w_min and dist < w_max]
                sample_number = len(ids)
                for idx in ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, bn_w_dist[idx], stats_list[idx], pseudo_cls], rmst)

        if isadapt:
            self.mem = torch.stack(self.memory.get_memory())
            
    def update_mem(self,logits):
        if self.mem is not None:
            self.memory.reset_memory()
            entropys = softmax_entropy(logits)
            confidences = get_confidence(logits) 
            for i, x_ins in enumerate(self.mem):
                logit = logits[i]
                pseudo_cls = logit.max(dim=0)[1]
                self.memory.add_instance([x_ins,entropys[i],confidences[i],pseudo_cls])
            if self.memory.get_occupancy() != self.memory.capacity:
                assert("Memory length doesnt match after update")
                    
        
    def reset_bn(self):
        for m in self.model.modules():
            if isinstance(m, MectaNorm2d):
                m.reset()

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # use this line if you want to reset the teacher model as well. Maybe you also
        # want to del self.model_ema first to save gpu memory.
        # self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
        #     copy_model_and_optimizer(self.model, self.optimizer)
    
    def switch_bn(self,adapt=True):
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                # if filter is not None and not filter(nm):
                #     continue
                m.requires_grad_(adapt)
                m.momentum = 1
                m.track_running_stats = adapt # update moving bn stat
                
    def forward_only(self, x):        
        outputs = self.model(x)
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(
            self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        outputs_emas = []
        to_aug = anchor_prob.mean(0) < 0.1
        if to_aug:
            for i in range(N):
                outputs_ = self.model_ema(self.transform(x)).detach()
                outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        return outputs_ema, outputs
        
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, optimizer, progress):
        outputs = self.model(x)
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(
            self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        outputs_emas = []
        to_aug = anchor_prob.mean(0) < 0.1
        if to_aug:
            for i in range(N):
                outputs_ = self.model_ema(self.transform(x)).detach()
                outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Augmentation-averaged Prediction
        # Student update
        loss = (softmax_entropy_cotta(
            outputs, outputs_ema)).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(
            ema_model=self.model_ema, model=self.model, alpha_teacher=0.999)
        # Stochastic restore
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    if self.device == 'cuda':
                        mask = (torch.rand(p.shape) < 0.001).float().cuda()
                    else:
                        mask = (torch.rand(p.shape) < 0.001).float()
                    with torch.no_grad():
                        p.data = self.model_state[f"{nm}.{npp}"] * \
                            mask + p * (1.-mask)
        return outputs_ema
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def adapt(self, optimizer, progress, mem,rmst,alginf=False):        
        ######################### For Adapt w memory #######################
        if mem is not None and self.memory is not None:
            outputs_withmem_list = []
            outputs_ema_list = []
            mem_dataset = TensorDataset(mem)  # mem이 torch.Tensor 타입이라고 가정합니다.
            mem_loader = DataLoader(mem_dataset, batch_size=self.memory.get_occupancy(), shuffle=True)
            for mem_batch in mem_loader:
                mem_batch = mem_batch[0] 
                outputs_withmem = self.model(mem_batch)
                outputs_withmem_list.append(outputs_withmem)
                
                # Teacher Prediction
                anchor_prob = torch.nn.functional.softmax(
                    self.model_anchor(mem_batch), dim=1).max(1)[0]
                standard_ema = self.model_ema(mem_batch)
                # Augmentation-averaged Prediction
                outputs_emas = []
                to_aug = anchor_prob.mean(0) < 0.1
                if to_aug:
                    for i in range(N):
                        outputs_ = self.model_ema(self.transform(mem_batch)).detach()
                        outputs_emas.append(outputs_)
                # Threshold choice discussed in supplementary
                if to_aug:
                    outputs_ema = torch.stack(outputs_emas).mean(0)
                else:
                    outputs_ema = standard_ema
                outputs_ema_list.append(outputs_ema)
            ############################################################
            outputs_ema = torch.cat(outputs_ema_list, dim=0)
            # 리스트에 있는 모든 결과를 하나의 텐서로 결합
            outputs = torch.cat(outputs_withmem_list, dim=0)
        else:
            assert('error on adapt without memory')
        # Student update
        loss = (softmax_entropy_cotta(outputs, outputs_ema)).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Teacher update
        self.model_ema = update_ema_variables(
            ema_model=self.model_ema, model=self.model, alpha_teacher=0.999)
        # Stochastic restore
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    if self.device == 'cuda':
                        mask = (torch.rand(p.shape) < 0.001).float().cuda()
                    else:
                        mask = (torch.rand(p.shape) < 0.001).float()
                    with torch.no_grad():
                        p.data = self.model_state[f"{nm}.{npp}"] * \
                            mask + p * (1.-mask)
                            
        TimeTracker.track(progress.get_meter('bp_time'))
        return outputs_ema
            

    @staticmethod
    def collect_params(model):
        """Collect all trainable parameters.

        Walk the model's modules and collect all parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        # print(nm, np)
        return params, names

    @staticmethod
    def configure_model(model):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what we update
        model.requires_grad_(False)
        # enable all trainable
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = True
                m.momentum = 1
                # m.running_mean = None
                # m.running_var = None
            else:
                m.requires_grad_(True)
        return model
    
    def reset_steps(self, new_steps):
        self.steps = new_steps
    
    def iobmn_get_bn_stats(self, compare_with_test=False,layer_t=0):
        return bn_iobmn_get_bn_stats(self, compare_with_test,layer_t)
            
    def iobmn_get_bn_stats_N(self, N, compare_with_test=False):
        return bn_iobmn_get_bn_stats_N(self, N, compare_with_test)
            
    def retrieve_bn_stats(self, outputs, isadapt,layer_t=0):
        return bn_retrieve_bn_stats(self, outputs, isadapt,layer_t)

    def retrieve_bn_stats_N(self, outputs, N, isadapt):
        return bn_retrieve_bn_stats_N(self, outputs, N, isadapt)

    def recalc_bn_stats(self, memtype,compare_with_test=False):
        return bn_recalc_bn_stats(self,memtype, compare_with_test)

    def check_bn_divergence(self, memtype,mu, sigma2):
        return bn_check_bn_divergence(self,memtype, mu, sigma2)



@torch.jit.script
def softmax_entropy_cotta(x: torch.Tensor, x_ema: torch.Tensor, symmetric: bool = False) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    if symmetric:
        return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)
    else:
        return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
    
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


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

def print_model_summary(module, input_tensor, trainonly=False, depth=0):
    indent = "    " * depth

    if trainonly:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in module.parameters())
    print(f"{indent}{module.__class__.__name__}: {num_params} params")

    for name, child in module.named_children():
        print_model_summary(child, input_tensor, trainonly, depth + 1)
    
    return

