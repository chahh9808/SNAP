"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np

from models.batch_norm import has_accum_bn_grad
from .base import AdaptableModule, collect_bn_params, configure_model
from utils.memory import NMemory, PBMemory

from utils.bn_utils import bn_iobmn_get_bn_stats, bn_iobmn_get_bn_stats_N, bn_retrieve_bn_stats, bn_retrieve_bn_stats_N, bn_recalc_bn_stats, bn_check_bn_divergence
from torch.utils.data import TensorDataset, DataLoader

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data

class SAR(AdaptableModule):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, e_margin, maxage, c_margin, steps=1, episodic=False, reset_constant_em=0.2,layer_t=0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        # Memory for effective adapting
        self.mem = None
        self.memory = None
        self.entrth_init = e_margin
        self.entrth = e_margin
        self.confth = c_margin
        self.maxage = maxage
        
        # domain centroid (mu,sigma2)
        self.mu_centr = None
        self.sigma2_centr = None

        # update memory's wdist if diverged too far
        self.wass_dist = [] # wass dist btw prev bn stats
        self.cnt = 0 # count of how many times wdist was reupdated

        self.bn_analysis = []

        # for wdist N layer normalization
        self.norm_beta = 0.01
        self.wasserstein_means_test = None
        self.wasserstein_vars_test = None
        
        self.layer_t=layer_t

    def forward(self, x, progress, isadapt, memtype, adst, rmst, mem_size, memreset, alginf=False):
        if self.episodic:
            self.reset()

        if isadapt:
            if self.memory is None:
                self.model.train()
                self.switch_bn(True,model=self.model)
                outputs, ema, reset_flag = self.forward_and_adapt_sar(x, self.model, self.optimizer, progress, self.reset_constant_em, self.ema)
                if reset_flag:
                    self.reset()
                self.ema = ema  # update moving average value of loss
                return outputs
                        
            self.model.eval()
            self.switch_bn(False,model=self.model)
            with torch.no_grad():
                outputs = self.model(x)

            wdists_test, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt,self.layer_t)
            self.check_bn_divergence(memtype,mu, sigma2)
            self.add_mem(x, memtype, adst, rmst, mem_size, isadapt, outputs, wdists_test, stats_list)
            #self.add_mem(x, memtype, adst, rmst, mem_size, isadapt, outputs)
            
            self.model.train()
            self.switch_bn(True,model=self.model)

            for _ in range(self.steps):
                _, ema, reset_flag = self.adapt(self.model, self.optimizer, progress, self.mem, outputs, rmst, self.ema)
                if reset_flag: 
                    self.reset()           
                self.ema = ema

            if memreset:
                self.mem = None

        else:
            # skip adapt, just inference
            self.model.eval()
            self.switch_bn(False,model=self.model)
            with torch.no_grad():
                outputs = self.model(x)

            wdists_test, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt,self.layer_t)
            self.check_bn_divergence(memtype,mu, sigma2)
            self.add_mem(x, memtype, adst, rmst, mem_size, isadapt, outputs, wdists_test, stats_list)
            #self.add_mem(x,memtype,adst,rmst,mem_size,isadapt,outputs)
            
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_all(self):
        self.reset()
        self.reset_steps(1)
        self.reset_bn()
        self.mem = None
        self.memory = None
        self.entrth = self.entrth_init

    def add_mem(self, x, memtype, adst, rmst, mem_size, isadapt, logits, wdists_test, stats_list):        
        if adst == 'basic':
            self.mem = x
            return 0
        
        entropys = softmax_entropy(logits)
        confidences = get_confidence(logits) 
        
        if self.memory is None:
            if memtype == 'normal':
                self.memory = NMemory(capacity=mem_size,num_class=logits.shape[-1],max_age_threshold=self.maxage)
            elif memtype == 'pb':
                self.memory = PBMemory(capacity=mem_size,num_class=logits.shape[-1],max_age_threshold=self.maxage)
            else:
                assert('NONE PREDEFINED MEMORY TYPE')
            
            for i, x_ins in enumerate(x):
                logit = logits[i]
                pseudo_cls = logit.max(dim=0)[1]
                self.memory.add_instance([x_ins, entropys[i], confidences[i], 0, wdists_test[i], stats_list[i], pseudo_cls], rmst)
             
        else:  
            if adst == 'all':
                for i, x_ins in enumerate(x):
                    logit = logits[i]
                    pseudo_cls = logit.max(dim=0)[1]
                    self.memory.add_instance([x_ins, entropys[i], confidences[i], 0, wdists_test[i], stats_list[i], pseudo_cls], rmst)
            elif adst == 'high_entr':
                hientro_ids = torch.where(entropys > self.entrth)[0].tolist()
                for idx in hientro_ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, wdists_test[idx], stats_list[idx], pseudo_cls], rmst)
            elif adst == 'low_entr':
                lowentro_ids = torch.where(entropys < self.entrth)[0].tolist()
                for idx in lowentro_ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, wdists_test[idx], stats_list[idx], pseudo_cls],rmst)
            elif adst == 'high_conf':
                hiconf_ids = torch.where((confidences > self.confth) & (confidences < 1))[0].tolist()
                for idx in hiconf_ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, wdists_test[idx], stats_list[idx], pseudo_cls],rmst)
    
        if isadapt:
            self.mem = torch.stack(self.memory.get_memory())

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_sar(self, x, model, optimizer, progress, reset_constant, ema):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        optimizer.zero_grad()
        # forward
        outputs = model(x)
        # adapt
        entropys = softmax_entropy(outputs) # no entropy filtering

        if has_accum_bn_grad(model):
            # first forward-backward pass
            loss = entropys.mean(0)
            loss.backward()
            optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)

            # second forward-backward pass
            entropys2 = softmax_entropy(model(x))
            loss_second_value = entropys2.clone().detach().mean(0)
            loss_second = entropys2.mean(0)

            if not np.isnan(loss_second.item()):
                ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

            # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
            loss_second.backward()
            optimizer.second_step(zero_grad=True)

        # perform model recovery
        reset_flag = False
        if ema is not None:
            if ema < reset_constant:
                # print("ema low, now reset the model")
                reset_flag = True

        return outputs, ema, reset_flag
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def adapt(self, model, optimizer, progress, mem, outputs, rmst, ema):
        """Adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # adapt
        batch_size = len(outputs)
        if mem is not None and self.memory is not None: 
            outputs_withmem_list = []
            mem_dataset = TensorDataset(mem)
            mem_loader = DataLoader(mem_dataset, batch_size=self.memory.get_occupancy(), shuffle=True)
            for mem_batch in mem_loader:
                mem_batch = mem_batch[0]
                outputs_withmem = model(mem_batch)
                outputs_withmem_list.append(outputs_withmem)
            outputs_withmem = torch.cat(outputs_withmem_list, dim=0)
            
        else: outputs_withmem = outputs
        loss = softmax_entropy(outputs_withmem).mean(0)
        
        if has_accum_bn_grad(model):
            # first forward-backward pass
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            if mem is not None and self.memory is not None: 
                outputs_withmem_list = []
                mem_dataset = TensorDataset(mem)
                mem_loader = DataLoader(mem_dataset, batch_size=self.memory.get_occupancy(), shuffle=True)
                for mem_batch in mem_loader:
                    mem_batch = mem_batch[0]
                    outputs_withmem = model(mem_batch)
                    outputs_withmem_list.append(outputs_withmem)
                outputs_withmem = torch.cat(outputs_withmem_list, dim=0)
                
            else: outputs_withmem = outputs
            loss_second = softmax_entropy(outputs_withmem).mean(0)

            if not np.isnan(loss_second.item()):
                ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

            # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
            loss_second.backward()
            optimizer.second_step(zero_grad=True)

            optimizer.zero_grad()

        # perform model recovery
        reset_flag = False
        if ema is not None:
            if ema < 0.2:
                # print("ema < 0.2, now reset the model")
                reset_flag = True

        return outputs, ema, reset_flag

    def iobmn_get_bn_stats(self, compare_with_test=False,layer_t=0):
        return bn_iobmn_get_bn_stats(self, compare_with_test,layer_t)
            
    def iobmn_get_bn_stats_N(self, N, compare_with_test=False):
        return bn_iobmn_get_bn_stats_N(self, N, compare_with_test)
            
    def retrieve_bn_stats(self, outputs, isadapt,layer_t=0):
        return bn_retrieve_bn_stats(self, outputs, isadapt,layer_t)

    def retrieve_bn_stats_N(self, outputs, N, isadapt):
        return bn_retrieve_bn_stats_N(self, outputs, N, isadapt)

    def recalc_bn_stats(self, memtype, compare_with_test=False):
        return bn_recalc_bn_stats(self, memtype,compare_with_test)

    def check_bn_divergence(self, memtype,mu, sigma2):
        return bn_check_bn_divergence(self, memtype,mu, sigma2)

    @staticmethod
    def collect_params(model):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """

        params = []
        names = []
        for nm, m in model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                # if filter is not None and not filter(nm):
                #     continue
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names
    
    @staticmethod
    def configure_model(model):
        """Configure model for use with SAR."""

        """
        # train mode, because SAR optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what SAR updates
        model.requires_grad_(False)
        # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        
        return model
        """
        return configure_model(model)

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
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def check_model(model):
    """Check model for compatability with SAR."""
    is_training = model.training
    assert is_training, "SAR needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "SAR needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "SAR should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "SAR needs normalization layer parameters for its optimization"
