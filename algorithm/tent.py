"""
Copyright to Tent Authors ICLR 2021 Spotlight
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import numpy as np

from models.batch_norm import has_accum_bn_grad
from .base import AdaptableModule, collect_bn_params, configure_model
from utils.cpu_mem_track import MemTracker
# from utils.gpu_mem_track import MemTracker

from utils.memory import NMemory, PBMemory
from utils.bn_utils import bn_iobmn_get_bn_stats, bn_iobmn_get_bn_stats_N, bn_retrieve_bn_stats, bn_retrieve_bn_stats_N, bn_recalc_bn_stats, bn_check_bn_divergence
from torch.utils.data import TensorDataset, DataLoader



class Tent(AdaptableModule):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, e_margin, maxage, c_margin, w_min, w_max, fishers=None, training_avg=None, training_var=None, steps=1, episodic=False,layer_t=0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
                
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
                
        # preliminary experiment
        self.time = 0 # time
        self.memory_stats = [] # adaptation sample analysis
        self.sample_number_stats = [] # threshold-abiding sample analysis
        self.bn_analysis = []

        # domain centroid (mu,sigma2)
        self.mu_centr = None
        self.sigma2_centr = None

        # update memory's wdist if diverged too far
        self.wass_dist = [] # wass dist btw prev bn stats
        self.cnt = 0 # count of how many times wdist was reupdated

        # custom threshold for wdist
        self.w_min = w_min
        self.w_max = w_max

        # for wdist N layer normalization
        self.norm_beta = 0.01
        self.wasserstein_means_test = None
        self.wasserstein_vars_test = None
        
        self.layer_t = layer_t

    def forward(self, x, progress, isadapt, memtype, adst, rmst, mem_size, memreset, alginf=False):
        if self.episodic:
            self.reset()
        
        # adaptation batch
        if isadapt:
            if self.memory is None:
                self.model.train()
                self.switch_bn(True,model=self.model)
                outputs = self.forward_and_adapt(x, self.model, self.optimizer, progress)
                return outputs
            
            # forward pass incoming batch
            self.model.eval()
            self.switch_bn(False,model=self.model)
            with torch.no_grad():
                outputs = self.model(x)
            
            # collect wasserstein metric of samples and add/remove samples from memory
            bn_w_dist, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt,self.layer_t)
            self.check_bn_divergence(memtype,mu, sigma2)
            sample_number = self.add_mem(x, memtype, adst, rmst, mem_size, isadapt, outputs, bn_w_dist, stats_list)

            # adapt using samples in memory
            self.model.train()
            self.switch_bn(True,model=self.model)

            for _ in range(self.steps):
                _ = self.adapt(self.model, self.optimizer, progress, self.mem, outputs, rmst)
            
            if memreset:
                self.mem = None
            
            # prelim experiment
            # self.time += 1
            # mem_stats = self.memory.get_memory_stats()
            # mem_stats_time = [(ent, conf, self.time, clss, wdist_test) for ent, conf, clss, wdist_test in mem_stats]
            # self.memory_stats.append(mem_stats_time)

        else:
            # inference only batch
            self.model.eval()
            self.switch_bn(False,model=self.model)
            with torch.no_grad():
                outputs = self.model(x)
            
            if adst == 'basic':  
                return outputs  
            # collect wasserstein metric of samples and add/remove samples from memory
            bn_w_dist, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt,self.layer_t)
            self.check_bn_divergence(memtype,mu, sigma2)
            sample_number = self.add_mem(x, memtype, adst, rmst, mem_size, isadapt, outputs, bn_w_dist, stats_list)
        
        # self.sample_number_stats.append((self.time, sample_number))

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_all(self):
        self.reset()
        self.reset_steps(1)
        self.reset_bn()
        self.mem = None
        self.memory = None
        self.entrth = self.entrth_init
    
    def add_mem(self, x, memtype, adst, rmst, mem_size, isadapt, logits, bn_w_dist, stats_list):
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
                    self.memory.add_instance([x_ins, entropys[i], confidences[i], 0, bn_w_dist[i], stats_list[i], pseudo_cls], rmst)
            elif adst == 'low_entr':
                lowentro_ids = torch.where(entropys < self.entrth)[0].tolist()
                sample_number = len(lowentro_ids)
                for idx in lowentro_ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, bn_w_dist[idx], stats_list[idx], pseudo_cls],rmst)
            elif adst == 'high_conf':
                hiconf_ids = torch.where(confidences > self.confth)[0].tolist()
                sample_number = len(hiconf_ids)
                for idx in hiconf_ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, bn_w_dist[idx], stats_list[idx], pseudo_cls],rmst)
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
     
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer, progress):
        MemTracker.track('Before inference')
        outputs = model(x)
        MemTracker.track('Do inference')
        loss = softmax_entropy(outputs).mean(0)
        
        if has_accum_bn_grad(model):
            loss.backward()
            MemTracker.track('Do backward')
            optimizer.step()
            optimizer.zero_grad()
            MemTracker.track('After optimizer.step')
        return outputs
       
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def adapt(self, model, optimizer, progress, mem, outputs, rmst):
        """Adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # adapt
        batch_size = len(outputs)
        if mem is not None and self.memory is not None: 
            outputs_withmem_list = []
            mem_dataset = TensorDataset(mem)
            mem_loader = DataLoader(mem_dataset, batch_size=self.memory.get_occupancy(), shuffle=True)
            MemTracker.track('Before inference')
            for mem_batch in mem_loader:
                mem_batch = mem_batch[0]
                outputs_withmem = model(mem_batch)
                MemTracker.track('Do inference')
                outputs_withmem_list.append(outputs_withmem)
            outputs_withmem = torch.cat(outputs_withmem_list, dim=0)

        else: 
            outputs_withmem = outputs
                    
        loss = softmax_entropy(outputs_withmem).mean(0)

        if has_accum_bn_grad(model):
            loss.backward()
            MemTracker.track('Do backward')
            optimizer.step()
            optimizer.zero_grad()
            MemTracker.track('After optimizer.step')

        return outputs
            
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

    def check_bn_divergence(self,memtype, mu, sigma2):
        return bn_check_bn_divergence(self,memtype, mu, sigma2)

    # returns memory stats of adaptation samples
    def return_prelim_list(self):
        all_stats = []

        for stats in self.memory_stats:
            entropies, confidences, times, class_idx, bn_w_dist, wdists_train = zip(*stats)
            entropies = np.array([e.cpu().item() if isinstance(e, torch.Tensor) else e for e in entropies])
            confidences = np.array([c.cpu().item() if isinstance(c, torch.Tensor) else c for c in confidences])
            class_idx = np.array(class_idx)
            times = np.array(times)
            bn_w_dist = np.array(bn_w_dist)
            wdists_train = np.array(wdists_train)
            all_stats.extend(list(zip(entropies, confidences, times, class_idx, bn_w_dist, wdists_train)))

        return all_stats

    # returns threshold-abiding sample number statistics
    def return_sample_list(self):
        return self.sample_number_stats

    # returns bn stats analysis list
    def return_bn_stats_list(self):
        return self.bn_analysis
    
    # returns how many times memory wdist was recalculated
    def return_recalc_bn_stats_count(self):
        return self.cnt
        
    @staticmethod
    def collect_params(model):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        return collect_bn_params(model)

    @staticmethod
    def configure_model(model):
        """Configure model for use with tent."""
        return configure_model(model)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.jit.script
def energy(x: torch.Tensor) -> torch.Tensor:
    """Energy calculation from logits."""
    temprature = 1.
    x = -(temprature*torch.logsumexp(x / temprature, dim=1))
    # if torch.rand(1) > 0.95:
    print('## energy ', x.mean(0).item())
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
