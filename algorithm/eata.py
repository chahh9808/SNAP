"""
Copyright to EATA ICML 2022 Authors, 2022.03.20
Based on Tent ICLR 2021 Spotlight. 
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from utils.memory import NMemory, PBMemory
import math
import torch.nn.functional as F
from models.batch_norm import has_accum_bn_grad, standard_bn_cxt
from .base import AdaptableModule
from .base import collect_bn_params, configure_model
from utils.cpu_mem_track import MemTracker
from torch.utils.data import TensorDataset, DataLoader

from utils.bn_utils import bn_iobmn_get_bn_stats, bn_iobmn_get_bn_stats_N, bn_retrieve_bn_stats, bn_retrieve_bn_stats_N, bn_recalc_bn_stats, bn_check_bn_divergence

class EATA(AdaptableModule):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, maxage, c_margin, fishers=None, fisher_alpha=2000.0, steps=1, episodic=False, e_margin=math.log(1000)/2-1, d_margin=0.05,layer_t=0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # log info
        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        
        self.num_batch_adapted = 0

        self.e_margin = e_margin # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.fishers = fishers # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = fisher_alpha # trade-off \beta for two losses (Eqn. 8) 

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
                outputs, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_eata(x, self.model, self.optimizer, self.fishers, self.e_margin, self.current_model_probs, fisher_alpha=self.fisher_alpha, num_samples_update=self.num_samples_update_2, d_margin=self.d_margin, progress=progress)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                if num_counts_2 > 0:
                    self.num_batch_adapted += 1
                self.reset_model_probs(updated_probs)
                return outputs
            
            MemTracker.track('Before inference') 
            self.model.eval()
            self.switch_bn(False,model=self.model)
            with torch.no_grad():
                outputs = self.model(x)
            # TimeTracker.track(progress.get_meter('fw_time'))   
            MemTracker.track('Do inference') 

            wdists_test, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt,self.layer_t)
            self.check_bn_divergence(memtype,mu, sigma2)
            self.add_mem(x, memtype, adst, rmst, mem_size, isadapt, outputs, wdists_test, stats_list)
            
            self.model.train()
            self.switch_bn(True,model=self.model)
            for _ in range(self.steps):
                _, num_counts_2, num_counts_1, updated_probs = self.adapt(self.optimizer,self.mem,self.fishers, self.e_margin, self.current_model_probs, fisher_alpha=self.fisher_alpha, num_samples_update=self.num_samples_update_2, d_margin=self.d_margin, progress=progress)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                if num_counts_2 > 0:
                    self.num_batch_adapted += 1
                self.reset_model_probs(updated_probs)

            if memreset:
                self.mem = None

        else:
            self.model.eval()
            self.switch_bn(False,model=self.model)
            with torch.no_grad():
                outputs = self.model(x)
            # TimeTracker.track(progress.get_meter('fw_time'))
            
            if adst == 'basic':  
                return outputs  

            wdists_test, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt,self.layer_t)
            self.check_bn_divergence(memtype,mu, sigma2)
            self.add_mem(x, memtype, adst, rmst, mem_size, isadapt, outputs, wdists_test, stats_list)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    def reset_all(self):
        self.reset()
        self.reset_steps(1)
        self.reset_model_probs(None)
        self.reset_bn()
        self.num_batch_adapted = 0
        self.mem = None
        self.memory = None
        self.entrth = self.entrth_init

    @staticmethod
    def configure_model(model):
        """Configure model for use with eata."""
        return configure_model(model)

    @staticmethod
    def collect_params(model):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        return collect_bn_params(model)
    
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
    def adapt(self, optimizer, mem,fishers, e_margin, current_model_probs, fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0,progress=None,alginf=False):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.

        Returns: 
            outputs - model outputs; 
            num_remained - the number of reliable and non-redundant samples; 
            num_reliable - the number of reliable samples;
            probs - the moving average  probability vector over all previous samples
        """
        
        ######################### For Adapt w memory #######################
        if mem is not None and self.memory is not None:
            outputs_withmem_list = []
            outputs_ema_list = []
            mem_dataset = TensorDataset(mem)  # Assumes mem is a torch.Tensor
            mem_loader = DataLoader(mem_dataset, batch_size=self.memory.get_occupancy(), shuffle=True)
            for mem_batch in mem_loader:
                mem_batch = mem_batch[0] 
                outputs_withmem = self.model(mem_batch)
                outputs_withmem_list.append(outputs_withmem)
        # forward
                outputs = torch.cat(outputs_withmem_list, dim=0)
        # adapt
                entropys = softmax_entropy(outputs)
                
                # print(f"Entropy mean: {entropys.mean(0)}, "
                #       f"Entropy std: {entropys.std(0)}")
                
                
                # filter unreliable samples
                filter_ids_1 = torch.where(entropys < e_margin)
                ids1 = filter_ids_1
                ids2 = torch.where(ids1[0]>-0.1)
                entropys = entropys[filter_ids_1] 
                # filter redundant samples
                if current_model_probs is not None: 
                    cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
                    filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
                    entropys = entropys[filter_ids_2]
                    ids2 = filter_ids_2
                    updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
                else:
                    updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
                coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
                # implementation version 1, compute loss, all samples backward (some unselected are masked)
                entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
                loss = entropys.mean(0)
                """
                # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
                # if x[ids1][ids2].size(0) != 0:
                #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
                """
                if fishers is not None:
                    ewc_loss = 0
                    for name, param in self.model.named_parameters():
                        if name in fishers:
                            ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
                    loss += ewc_loss
                if mem_batch[ids1][ids2].size(0) != 0:
                    if has_accum_bn_grad(self.model):
                        loss.backward()
                        optimizer.step()
                optimizer.zero_grad()
                # TimeTracker.track(progress.get_meter('bp_time'))
        # print(f"Loss: {loss}, "
        #       f"num of fil1: {16-filter_ids_1[0].size(0)}, "
        #       f"num of fil2: {filter_ids_1[0].size(0)-entropys.size(0)}")
        return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs

    def iobmn_get_bn_stats(self, compare_with_test=False,layer_t=0):
        return bn_iobmn_get_bn_stats(self, compare_with_test,layer_t)
            
    def iobmn_get_bn_stats_N(self, N, compare_with_test=False):
        return bn_iobmn_get_bn_stats_N(self, N, compare_with_test)
            
    def retrieve_bn_stats(self, outputs, isadapt,layer_t=0):
        return bn_retrieve_bn_stats(self, outputs, isadapt,layer_t)

    def retrieve_bn_stats_N(self, outputs, N, isadapt):
        return bn_retrieve_bn_stats_N(self, outputs, N, isadapt)

    def recalc_bn_stats(self, memtype, compare_with_test=False):
        return bn_recalc_bn_stats(self, memtype, compare_with_test)

    def check_bn_divergence(self, memtype, mu, sigma2):
        return bn_check_bn_divergence(self,memtype, mu, sigma2)




@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_eata(x, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0,progress=None):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.

    Returns: 
        outputs - model outputs; 
        num_remained - the number of reliable and non-redundant samples; 
        num_reliable - the number of reliable samples;
        probs - the moving average  probability vector over all previous samples
    """
    # forward
    outputs = model(x)
    # TimeTracker.track(progress.get_meter('fw_time'))
    # adapt
    entropys = softmax_entropy(outputs)
    
    # print(f"Entropy mean: {entropys.mean(0)}, "
    #       f"Entropy std: {entropys.std(0)}")
    
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0]>-0.1)
    entropys = entropys[filter_ids_1] 
    # filter redundant samples
    if current_model_probs is not None: 
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        ids2 = filter_ids_2
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    # implementation version 1, compute loss, all samples backward (some unselected are masked)
    entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
    loss = entropys.mean(0)
    """
    # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
    # if x[ids1][ids2].size(0) != 0:
    #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
    """
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss
    if x[ids1][ids2].size(0) != 0:
        if has_accum_bn_grad(model):
            loss.backward()
            optimizer.step()
    optimizer.zero_grad()
    # TimeTracker.track(progress.get_meter('bp_time'))
    # print(f"Loss: {loss}, "
    #       f"num of fil1: {16-filter_ids_1[0].size(0)}, "
    #       f"num of fil2: {filter_ids_1[0].size(0)-entropys.size(0)}")
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


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
    """Check model for compatability with eata."""
    is_training = model.training
    assert is_training, "eata needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "eata needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "eata should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "eata needs normalization for its optimization"


def compute_fishers(params, subnet, fisher_loader, device):
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    for iter_, (images, targets) in enumerate(fisher_loader, start=1):
        images = images.to(device)
        # targets = targets.to(device)
        with standard_bn_cxt(subnet):
            outputs = subnet(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
        for name, param in subnet.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    print("compute fisher matrices finished")
    del ewc_optimizer
    return fishers

@torch.jit.script
def get_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Get confidence from logits."""
    probabilities = logits.softmax(1)
    confidence, _ = torch.max(probabilities, dim=1)
    return confidence