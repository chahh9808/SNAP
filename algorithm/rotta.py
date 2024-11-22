import torch
import torch.nn as nn
import torch.jit
import numpy as np

from copy import deepcopy
from models.batch_norm import has_accum_bn_grad
from .base import AdaptableModule, collect_bn_params, configure_model
from utils.cpu_mem_track import MemTracker
# from utils.gpu_mem_track import MemTracker
from utils.latency_track import TimeTracker

from utils.memory import NMemory, PBMemory
from utils.custom_transforms import get_tta_transforms
from utils.bn_utils import bn_iobmn_get_bn_stats, bn_iobmn_get_bn_stats_N, bn_retrieve_bn_stats, bn_retrieve_bn_stats_N, bn_recalc_bn_stats, bn_check_bn_divergence
from torch.utils.data import TensorDataset, DataLoader

class RoTTA(AdaptableModule):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, e_margin, maxage, c_margin, w_min, w_max, input_size, fishers=None, training_avg=None, training_var=None):
        super().__init__()
        self.model_student = model # student model
        self.model = self.build_ema(self.model_student) # teacher ema
        self.optimizer = optimizer
        self.transform = get_tta_transforms(input_size)
        self.nu = 0.001

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
            
        # Memory for effective adapting
        self.mem = None
        self.mem_age = None
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

    def forward(self, x, progress, isadapt, memtype, adst, rmst, mem_size, memreset, alginf=False):        
        # adaptation batch
        if isadapt:
            if self.memory is None:
                self.model.train()
                self.switch_bn(True,model=self.model)
                outputs = self.forward_and_adapt(x, self.model, self.optimizer, progress)
                return outputs
            
            # forward pass incoming batch
            self.model_student.eval()
            self.model.eval()
            self.switch_bn(False,model=self.model_student)
            self.switch_bn(False,model=self.model)

            with torch.no_grad():
                outputs = self.model(x)
            
            # collect wasserstein metric of samples and add/remove samples from memory
            wdists_test, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt)
            self.check_bn_divergence(memtype,mu, sigma2)
            sample_number = self.add_mem(x, memtype, adst, rmst, mem_size, isadapt, outputs, wdists_test, stats_list)

            # adapt using samples in memory
            self.model_student.train()
            self.model.train()
            self.switch_bn(True,model=self.model_student)
            self.switch_bn(True,model=self.model)

            self.adapt(self.model, self.optimizer, progress, self.mem, self.mem_age, outputs, rmst)
            
            if memreset:
                self.mem = None
                self.mem_age = None
            
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
                
            # collect wasserstein metric of samples and add/remove samples from memory
            wdists_test, stats_list, mu, sigma2 = self.retrieve_bn_stats(outputs, isadapt)
            self.check_bn_divergence(memtype,mu, sigma2)
            sample_number = self.add_mem(x, memtype, adst, rmst, mem_size, isadapt, outputs, wdists_test, stats_list)
        
        self.sample_number_stats.append((self.time, sample_number))

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_all(self):
        self.reset()
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
                sample_number = len(x)
                logit = logits[i]
                pseudo_cls = logit.max(dim=0)[1]
                self.memory.add_instance([x_ins, entropys[i], confidences[i], 0, wdists_test[i], stats_list[i], pseudo_cls], rmst)
                
        
        else:
            if adst == 'all':
                sample_number = len(x)
                for i, x_ins in enumerate(x):
                    logit = logits[i]
                    pseudo_cls = logit.max(dim=0)[1]
                    self.memory.add_instance([x_ins, entropys[i], confidences[i], 0, wdists_test[i], stats_list[i], pseudo_cls], rmst)
            elif adst == 'low_entr':
                lowentro_ids = torch.where(entropys < self.entrth)[0].tolist()
                sample_number = len(lowentro_ids)
                for idx in lowentro_ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, wdists_test[idx], stats_list[idx], pseudo_cls],rmst)
            elif adst == 'high_conf':
                hiconf_ids = torch.where((confidences > self.confth) & (confidences < 1))[0].tolist()
                sample_number = len(hiconf_ids)
                for idx in hiconf_ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, wdists_test[idx], stats_list[idx], pseudo_cls],rmst)
            elif adst == 'wdist_custom': # custom min/max threshold from input
                w_min = self.w_min
                w_max = self.w_max
                ids = [i for i, dist in enumerate(wdists_test) if dist > w_min and dist < w_max]
                sample_number = len(ids)
                for idx in ids:
                    pseudo_cls = logits[idx].max(dim=0)[1]
                    self.memory.add_instance([x[idx], entropys[idx], confidences[idx], 0, wdists_test[idx], stats_list[idx], pseudo_cls], rmst)
            
        if isadapt:
            self.mem = torch.stack(self.memory.get_memory())
            self.mem_age = torch.tensor(self.memory.get_memory_age())
            
        return sample_number
     
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer, progress): 
        #TimeTracker.set_timestamp()
        outputs = self.model(x)
        #TimeTracker.track(progress.get_meter('fw_time'))

        #TimeTracker.set_timestamp()
        strong_sup_aug = self.transform(x)
        #TimeTracker.track(progress.get_meter('transform'))

        #ema_sup_out = self.model(x)
        ema_sup_out = outputs
        #TimeTracker.set_timestamp()
        stu_sup_out = self.model_student(strong_sup_aug)
        #TimeTracker.track(progress.get_meter('fw_time_2'))

        l_sup = (softmax_entropy_rotta(stu_sup_out, ema_sup_out)).mean()

        loss = l_sup
        
        if has_accum_bn_grad(model):
            #TimeTracker.set_timestamp()
            loss.backward()
            #TimeTracker.track(progress.get_meter('bp_time'))
            #TimeTracker.set_timestamp()
            optimizer.step()
            #TimeTracker.track(progress.get_meter('optstep_time'))
            optimizer.zero_grad()
        
        #TimeTracker.set_timestamp()
        self.update_ema_variables(self.model, self.model_student, self.nu)
        #TimeTracker.track(progress.get_meter('update_ema'))

        return outputs
       
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def adapt(self, model, optimizer, progress, mem, mem_age, outputs, rmst):
        """Adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # adapt
        batch_size = len(outputs)
        if mem is not None and self.memory is not None: 
            ema_sup_out_list = []
            stu_sup_out_list = []
            instance_weight_list = []

            mem_dataset = TensorDataset(mem, mem_age)
            mem_loader = DataLoader(mem_dataset, batch_size=self.memory.get_occupancy(), shuffle=True)
            for mem_batch, age_batch in mem_loader:
                strong_sup_aug = self.transform(mem_batch)

                ema_sup_out = self.model(mem_batch)
                stu_sup_out = self.model_student(strong_sup_aug)

                instance_weight = timeliness_reweighting(age_batch)

                ema_sup_out_list.append(ema_sup_out)
                stu_sup_out_list.append(stu_sup_out)
                instance_weight_list.append(instance_weight)

            ema_sup_out = torch.cat(ema_sup_out_list, dim=0)
            stu_sup_out = torch.cat(stu_sup_out_list, dim=0)
            instance_weight = torch.cat(instance_weight_list, dim=0)
            instance_weight = instance_weight.to('cuda')

            l_sup = (softmax_entropy_rotta(stu_sup_out, ema_sup_out) * instance_weight).mean()

        else: 
            outputs_withmem = outputs
            l_sup = softmax_entropy(outputs_withmem).mean(0)
        
        loss = l_sup

        if has_accum_bn_grad(model):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.update_ema_variables(self.model, self.model_student, self.nu)

        return outputs
            
    def iobmn_get_bn_stats(self, compare_with_test=False):
        return bn_iobmn_get_bn_stats(self, compare_with_test)
            
    def iobmn_get_bn_stats_N(self, N, compare_with_test=False):
        return bn_iobmn_get_bn_stats_N(self, N, compare_with_test)
            
    def retrieve_bn_stats(self, outputs, isadapt):
        return bn_retrieve_bn_stats(self, outputs, isadapt)

    def retrieve_bn_stats_N(self, outputs, N, isadapt):
        return bn_retrieve_bn_stats_N(self, outputs, N, isadapt)

    def recalc_bn_stats(self, memtype,compare_with_test=False):
        return bn_recalc_bn_stats(self,memtype, compare_with_test)

    def check_bn_divergence(self, memtype,mu, sigma2):
        return bn_check_bn_divergence(self,memtype, mu, sigma2)
    
    # returns how many times memory wdist was recalculated
    def return_recalc_bn_stats_count(self):
        return self.cnt
        
    @staticmethod
    def collect_params(model):
        return collect_bn_params(model)

    @staticmethod
    def configure_model(model):
        return configure_model(model)

    @staticmethod
    def build_ema(model):
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model
    
    def iobmn_get_bn_stats(self, compare_with_test=False):
        return bn_iobmn_get_bn_stats(self, compare_with_test)
            
    def iobmn_get_bn_stats_N(self, N, compare_with_test=False):
        return bn_iobmn_get_bn_stats_N(self, N, compare_with_test)
            
    def retrieve_bn_stats(self, outputs, isadapt):
        return bn_retrieve_bn_stats(self, outputs, isadapt)

    def retrieve_bn_stats_N(self, outputs, N, isadapt):
        return bn_retrieve_bn_stats_N(self, outputs, N, isadapt)

    def recalc_bn_stats(self, memtype,compare_with_test=False):
        return bn_recalc_bn_stats(self, memtype,compare_with_test)

    def check_bn_divergence(self,memtype, mu, sigma2):
        return bn_check_bn_divergence(self,memtype, mu, sigma2)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

# for rotta specific
@torch.jit.script
def softmax_entropy_rotta(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))

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