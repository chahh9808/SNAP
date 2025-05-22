import torch
import torch.nn as nn
from collections import defaultdict

from models.batch_norm import MectaNorm2d

from utils.iobmn import convert_iobmn

import numpy as np

class AdaptableModule(nn.Module):
    """Module that can adapt model at test time."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def reset_all(self):
        raise NotImplementedError()

    def reset_bn(self):
        for m in self.model.modules():
            if isinstance(m, MectaNorm2d):
                m.reset()
                
    def switch_bn(self,adapt=True,model=None):
        # filter = parse_filter(model, filter)
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(adapt)
                m.momentum = 1
                m.track_running_stats = adapt # update moving bn stat
            elif isinstance(m, nn.LayerNorm):
                # if filter is not None and not filter(nm):
                #     continue
                # LayerNorm은 `momentum`이나 `track_running_stats` 속성이 없으므로 requires_grad만 조정
                m.requires_grad_(adapt)
                
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
        print("  배치 평균:", bn_layer.running_mean[0])
        print("  배치 분산:", bn_layer.running_var[0])
        print("  gamma:", bn_layer.weight)
        print("  beta:", bn_layer.bias)
    
    def print_first_ln_layer_stats(self):
        ln_layer = None
        for layer in self.model.modules():
            if isinstance(layer, nn.LayerNorm):
                ln_layer = layer
                break

        if ln_layer is None:
            print("모델에 LayerNorm 레이어가 없습니다.")
            return

        print("첫 번째 LayerNorm 레이어의 통계:")
        if ln_layer.elementwise_affine:
            print("  gamma (weight):", ln_layer.weight)
            print("  beta (bias):", ln_layer.bias)
        else:
            print("  이 LayerNorm 레이어는 affine 파라미터를 사용하지 않습니다.")

    @staticmethod
    def collect_params(model):
        """Collect parameters to update."""
        raise NotImplementedError()

    @staticmethod
    def configure_model(model):
        """Configure model, e.g., training status, gradient requirements."""
        raise NotImplementedError()


def configure_model(model):
    """Configure model for use with eata."""    
    # train mode, because eata optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what eata updates
    model.requires_grad_(False)
    # configure norm for eata updates: enable grad + force batch statisics
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # if filter is not None and not filter(nm):
            #     continue
            # print(f" # require grad for {nm}")
            m.requires_grad_(True)

            # store training first and second order statistics in each BN layer
            m.training_mean = m.running_mean.detach().cpu().clone().numpy()
            m.training_var = m.running_var.detach().cpu().clone().numpy()

            # force use of batch stats in train and eval modes
            m.track_running_stats = True
            m.momentum = 1 # force to use adapt-batch's statistics only
            # m.running_mean = None
            # m.running_var = None
        if isinstance(m, nn.LayerNorm):
            # if filter is not None and not filter(nm):
            #     continue
            # Enable gradient computation for the LayerNorm module
            m.requires_grad_(True)

    return model


def collect_bn_params(model):
    params = defaultdict(list)
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d,)) or isinstance(m, (nn.LayerNorm,)):
            # if filter is not None and not filter(nm):
            #     continue
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params['affine'].append(p)
                    names.append(f"{nm}.{np}")
            # print(f' train module: {nm}')
    return params, names


# def parse_filter(model, filter):
#     if filter is not None:
#         if isinstance(filter, str):
#             from models.eata_resnet import ResNet
#             assert filter.startswith('layer')
#             assert isinstance(
#                 model, ResNet), f"Unsupported model arch: {type(model)}"
#             start, end = filter[len('layer'):].split('-')
#             start, end = int(start), int(end)
#             def filter(nm): return start <= int(nm[len('layer'):].split(
#                 '.')[0]) <= end if nm.startswith('layer') else start == 0
#         elif isinstance(filter, int):
#             n_layer = filter
#             cnt = 0
#             layer_names = []
#             for n, m in model.named_modules():
#                 if 'downsample' in n:
#                     continue
#                 if isinstance(m, nn.BatchNorm2d):
#                     cnt += 1
#                     layer_names.append(n)
#             if n_layer > 0:
#                 layer_names = layer_names[-n_layer:]
#             else:
#                 layer_names = layer_names[:-n_layer]
#             assert len(layer_names) == abs(n_layer)
#             def filter(n): return n in layer_names
#         else:
#             raise RuntimeError(f"Unknown filter: {filter}")
#     return filter
