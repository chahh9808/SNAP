import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def convert_iobmn(module, iobmn_k=4, iobmn_s=1, use_tb=False,use_mtb=False, **kwargs):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Replace BatchNorm2d with SparseAdaptationAwareBatchNorm2d
            SAABN = SparseAdaptationAwareBatchNorm2d
            converted_bn = SAABN(
                num_channels=child.num_features,
                k=iobmn_k,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                s=iobmn_s,
                use_tb=use_tb,
                use_mtb=use_mtb,
            )
            converted_bn._bn = copy.deepcopy(child)
            setattr(module, name, converted_bn)  # Replace the original module

        else:
            # Recursively apply to children
            convert_iobmn(child, iobmn_k=iobmn_k, iobmn_s=iobmn_s, use_tb=use_tb, **kwargs)

    return module

class SparseAdaptationAwareBatchNorm2d(nn.Module):
    def __init__(self, num_channels, k=4, eps=1e-5, momentum=0.1, affine=True,s=0.1,use_tb=False,use_mtb=False):
        super(SparseAdaptationAwareBatchNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.k=k
        self.affine = affine
        self._bn = nn.BatchNorm2d(num_channels, eps=eps,
                                  momentum=momentum, affine=affine)
        self.s=s
        self.use_tb=use_tb
        self.use_mtb=use_mtb

        self.mu_batch = None
        self.sigma2_batch = None
        
        self.mu_cur = None
        self.sigma2_cur = None

        self.mu_test = None
        self.sigma2_test = None
        self.momentum = 0.1
    
    def _weighted_softshrink(self, x, lbd, weight_func):
        weighted_x_p = weight_func(F.relu(x - lbd, inplace=True))
        weighted_x_n = weight_func(F.relu(-(x + lbd), inplace=True))
        y = weighted_x_p - weighted_x_n
        return y
    
    def weight_func(self,x):
        return self.s * x
        
    def forward(self, x):
        b, c, h, w = x.size()
        self.sigma2_batch, self.mu_batch = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True) #INSTANCE
        mu = self._bn.running_mean.view(1, c, 1, 1) #MEMORY
        sigma2 = self._bn.running_var.view(1, c, 1, 1)

        # keep moving average of test domain statistics
        self.sigma2_cur, self.mu_cur = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
        # Detach the test statistics to prevent them from being part of the computational graph
        if self.mu_test is None and self.sigma2_test is None:
            self.mu_test = self.mu_cur.detach()  # detach here
            self.sigma2_test = self.sigma2_cur.detach()  # detach here
        else:
            self.mu_test = (1 - self.momentum) * self.mu_test.detach() + self.momentum * self.mu_cur.detach()  # detach mu_test and mu_cur
            self.sigma2_test = (1 - self.momentum) * self.sigma2_test.detach() + self.momentum * self.sigma2_cur.detach()  # detach sigma2_test and sigma2_cur
            

        if self.training:
            return self._bn(x)
        with torch.no_grad():
            if self.use_tb:
                mu_adj, sigma2_adj = self.mu_cur, self.sigma2_cur
                x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
            elif self.use_mtb:
                mu_adj, sigma2_adj = self.mu_test, self.sigma2_test
                x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
            else:
                sigma2_b, mu_b = self.sigma2_cur, self.mu_cur 
                s_mu = torch.sqrt((sigma2 + self.eps) / (h * w))
                s_sigma2 = (sigma2 + self.eps) * np.sqrt(2 / (h * w - 1))

                mu_adj = mu + self._weighted_softshrink(mu_b - mu, self.k * s_mu, self.weight_func)
                sigma2_adj = sigma2 + self._weighted_softshrink(sigma2_b - sigma2, self.k * s_sigma2, self.weight_func)

                mu_adj = torch.where(mu_b < mu, torch.max(mu_adj, mu_b), torch.min(mu_adj, mu_b))          
                sigma2_adj = torch.where(sigma2_b < sigma2, torch.max(sigma2_adj, sigma2_b), torch.min(sigma2_adj, sigma2_b))
                sigma2_adj = F.relu(sigma2_adj) #non negative
        
                x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
                
            if self.affine:
                weight = self._bn.weight.view(c, 1, 1)
                bias = self._bn.bias.view(c, 1, 1)
                x_n = x_n * weight + bias
            return x_n

# convert_iobmn modified for Vision Transformers (handling LayerNorm)
def convert_iobmn_vit(module, iobmn_k=25, iobmn_s=0.1, use_tb=False, use_mtb=False, **kwargs):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            # Replace LayerNorm with SparseAdaptationAwareLayerNorm
            SAALN = SparseAdaptationAwareLayerNorm
            converted_ln = SAALN(
                num_channels=child.normalized_shape[0],
                k=iobmn_k,
                eps=child.eps,
                affine=child.elementwise_affine,
                s=iobmn_s,
                use_tb=use_tb,
                use_mtb=use_mtb,
            )
            converted_ln._ln = copy.deepcopy(child)
            setattr(module, name, converted_ln)  # Replace the original module

        else:
            # Recursively apply to children
            convert_iobmn_vit(child, iobmn_k=iobmn_k, iobmn_s=iobmn_s, use_tb=use_tb, **kwargs)

    return module

class SparseAdaptationAwareLayerNorm(nn.Module):
    def __init__(self, num_channels, k=3.0, eps=1e-5, affine=True, s=0.1, use_tb=False, use_mtb=False):
        super(SparseAdaptationAwareLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.k = k
        self.affine = affine
        self.s = s
        self.use_tb = use_tb
        self.use_mtb = use_mtb
        self._ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)
        
        self.mu_memory = None
        self.sigma2_memory = None
        
        self.mu_batch = None
        self.sigma2_batch = None
        self.mu_cur = None
        self.sigma2_cur = None
        self.mu_prev = None
        self.sigma2_prev = None
        self.mu_test = None
        self.sigma2_test = None
        self.momentum = 0.1

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return self.s * y

    def _weighted_softshrink(self, x, lbd, weight_func):
        weighted_x_p = weight_func(F.relu(x - lbd, inplace=True))
        weighted_x_n = weight_func(F.relu(-(x + lbd), inplace=True))
        y = weighted_x_p - weighted_x_n
        return y

    def weight_func(self, x):
        return self.s * x
    
    def update_prev(self):
        self.mu_prev = self.mu_test
        self.sigma2_prev = self.sigma2_test

    def forward(self, x):
        # LayerNorm expects normalization over the last dimension, so adjust accordingly
        b, t, c = x.size()  # assuming input is [batch, tokens, channels]
        ############# Ver 1 - use layer norm value #############
        # Calculate the mean and variance over the last dimension (the channels for LayerNorm)
        self.sigma2_batch, self.mu_batch = torch.var_mean(x, dim=-1, keepdim=True, unbiased=True)
        # Keep moving averages of test domain statistics
        self.sigma2_cur, self.mu_cur = torch.var_mean(x, dim=[0,2], keepdim=True, unbiased=True)

        if self.mu_test is None and self.sigma2_test is None:
            # Initialize the test statistics
            self.mu_test = self.mu_cur.detach()
            self.sigma2_test = self.sigma2_cur.detach()
            self.mu_prev = self.mu_test.detach()
            self.sigma2_prev = self.sigma2_test.detach()
        else:
            # Exponentially moving average of the test statistics
            self.mu_test = (1 - self.momentum) * self.mu_test.detach() + self.momentum * self.mu_cur.detach()
            self.sigma2_test = (1 - self.momentum) * self.sigma2_test.detach() + self.momentum * self.sigma2_cur.detach()

        if self.training or self.use_tb:
            self.mu_memory = self.mu_cur.detach()
            self.sigma2_memory = self.sigma2_cur.detach()
            return self._ln(x)
        elif self.use_mtb:
            sigma2_b, mu_b = self.sigma2_cur, self.mu_cur 
        else:
            mu = self.mu_memory.view(1,t,1) #MEMORY
            sigma2 = self.sigma2_memory.view(1,t,1)
            sigma2_b, mu_b = self.sigma2_batch, self.mu_batch 
            s_mu = torch.sqrt((sigma2 + self.eps) / c)
            s_sigma2 = (sigma2 + self.eps) * np.sqrt(2 / (c - 1))

            # Adjust memory statistics according to batch statistics
            mu_adj = mu + self._weighted_softshrink(mu_b - mu, self.k * s_mu, self.weight_func)
            sigma2_adj = sigma2 + self._weighted_softshrink(sigma2_b - sigma2, self.k * s_sigma2, self.weight_func)

            # Clip mu_adj and sigma2_adj by mu_b and sigma2_b
            mu_adj = torch.where(mu_b < mu, torch.max(mu_adj, mu_b), torch.min(mu_adj, mu_b))
            sigma2_adj = torch.where(sigma2_b < sigma2, torch.max(sigma2_adj, sigma2_b), torch.min(sigma2_adj, sigma2_b))
            sigma2_adj = F.relu(sigma2_adj)

            x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        if self.affine:
            weight = self._ln.weight
            bias = self._ln.bias
            x_n = x_n * weight + bias

        return x_n
