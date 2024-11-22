"""
Copyright to Tent Authors ICLR 2021 Spotlight
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from collections import defaultdict

from models.batch_norm import has_accum_bn_grad
from .base import AdaptableModule, collect_bn_params, configure_model
from utils.cpu_mem_track import MemTracker
# from utils.gpu_mem_track import MemTracker
from utils.latency_track import TimeTracker

import math
from utils.memory import NMemory, PBMemory
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

class BNSTATS(AdaptableModule):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, e_margin, maxage, fishers=None, training_avg=None, training_var=None, steps=1, episodic=False):
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
        self.memoutput = None
        self.prev_memoutput = None
        self.entrth_init = e_margin
        self.entrth = e_margin
        
        self.maxage = maxage
        
        # domain centroid (mu,sigma2)
        self.mu_centr = None
        self.sigma2_centr = None
        
        self.global_entropys = []
        self.global_confidences = []

    def forward(self, x, progress, isadapt, memtype, adst, rmst, mem_size, memreset,alginf=False):
        if self.episodic:
            self.reset()
        # self.add_mem(x,strategy,isadapt)
        if isadapt:
            if self.memory is None:
                self.model.train()
                self.switch_bn(True,model=self.model)
                outputs = self.forward_and_adapt(x,self.model,self.optimizer,progress)
                return outputs
            
            # MemTracker.track('Before inference') 
            self.model.eval()
            self.switch_bn(False,model=self.model)
            with torch.no_grad():
                outputs = self.model(x)
            # TimeTracker.track(progress.get_meter('fw_time'))   
            # MemTracker.track('Do inference') 
            self.add_mem(x,memtype,adst,rmst,mem_size,isadapt,outputs)
            
            self.model.train()
            self.switch_bn(True,model=self.model)
            for _ in range(self.steps):
                _ = self.adapt(self.model,self.optimizer,progress,self.mem,outputs,rmst,alginf)            
            if memreset:
                self.mem = None
        else:
            self.model.eval()
            self.switch_bn(False,model=self.model)
            with torch.no_grad():
                outputs = self.model(x)
            # TimeTracker.track(progress.get_meter('fw_time'))
            self.add_mem(x,memtype,adst,rmst,mem_size,isadapt,outputs)
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
        self.memoutput = None
        self.prev_memoutput = None
        self.entrth = self.entrth_init
        self.global_entropys = []
        self.global_confidences = []
    
    def add_mem(self,x,memtype,adst,rmst,mem_size,isadapt,logits):
        # bn_size = len(x)
        # mem_size = mem_size
        
        if adst == 'basic':
            self.mem = x
            return 0
        
        if self.memory is None:
            if memtype == 'normal':
                self.memory = NMemory(capacity=mem_size,num_class=logits.shape[-1])
            elif memtype == 'pb':
                self.memory = PBMemory(capacity=mem_size,num_class=logits.shape[-1],max_age_threshold=self.maxage)
            else:
                assert('NONE PREDEFINED MEMORY TYPE')
                
        entropys = softmax_entropy(logits)
        confidences = get_confidence(logits)        
        if adst == 'all' or self.mem == None:
            for i, x_ins in enumerate(x):
                logit = logits[i]
                pseudo_cls = logit.max(dim=0)[1]
                self.memory.add_instance([x_ins,entropys[i],confidences[i],pseudo_cls],rmst)
        elif adst == 'high_entr':
            hientro_ids = torch.where(entropys > self.entrth)[0].tolist()
            hientro_ids = torch.where(confidences > 0.5)[0].tolist()
            for idx in hientro_ids:
                pseudo_cls = logits[idx].max(dim=0)[1]
                self.memory.add_instance([x[idx],entropys[idx],confidences[idx],pseudo_cls],rmst)
        elif adst == 'low_entr' and isadapt:
            lowentro_ids = torch.where(entropys < self.entrth)[0].tolist()
            for idx in lowentro_ids:
                pseudo_cls = logits[idx].max(dim=0)[1]
                self.memory.add_instance([x[idx],entropys[idx],confidences[idx],pseudo_cls],rmst)
        
        # self.global_entropys.extend(entropys.cpu().tolist())
        # self.global_confidences.extend(confidences.cpu().tolist())
            
        if isadapt:
            self.mem = torch.stack(self.memory.get_memory())
            # print(len(self.mem))      
            # self.memory.print_class_dist()
        
     
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer, progress):
        outputs = model(x)
        loss = softmax_entropy(outputs).mean(0)
        
        if has_accum_bn_grad(model):
            loss.backward()
            # MemTracker.track('Do backward')
            optimizer.step()
            optimizer.zero_grad()
            # MemTracker.track('After optimizer.step')
            # TimeTracker.track(progress.get_meter('bp_time'))
        return outputs
       
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def adapt(self, model, optimizer, progress, mem, outputs,rmst,alginf=False):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # MemTracker.track('Before inference')
        # # forward
        # outputs = model(x)
        # TimeTracker.track(progress.get_meter('fw_time'))
        # MemTracker.track('Do inference')
        # adapt
        batch_size = len(outputs)
        if mem is not None and self.memory is not None: 
            outputs_withmem_list = []
            mem_dataset = TensorDataset(mem)
            mem_loader = DataLoader(mem_dataset, batch_size=self.memory.get_occupancy(), shuffle=True)
            for mem_batch in mem_loader:
                mem_batch = mem_batch[0]  # TensorDataset으로부터 얻은 배치는 튜플 형태이므로, 실제 데이터를 추출합니다.
                outputs_withmem = model(mem_batch)
                outputs_withmem_list.append(outputs_withmem)
            # 리스트에 있는 모든 결과를 하나의 텐서로 결합
            outputs_withmem = torch.cat(outputs_withmem_list, dim=0)
            # outputs_withmem = model(mem)
            # print(outputs_withmem.shape)
            self.prev_memoutput = self.memoutput
            self.memoutput = outputs_withmem
            
            # if rmst != 'RAND' and upmem:
            #     self.update_mem(outputs_withmem)
        else: outputs_withmem = outputs
        loss = softmax_entropy(outputs_withmem).mean(0)
        
        # for group in optimizer.param_groups:
        #     for param in group['params']:
        #         # 모델의 파라미터를 순회하며 optimizer에 포함된 파라미터의 이름 찾기
        #         for name, p in model.named_parameters():
        #             if p is param:
        #                 print(f"Updating parameter: {name}")
        
        if has_accum_bn_grad(model):
            loss.backward()
            # MemTracker.track('Do backward')
            optimizer.step()
            optimizer.zero_grad()
            # MemTracker.track('After optimizer.step')
            # TimeTracker.track(progress.get_meter('bp_time'))
        return outputs
    
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
