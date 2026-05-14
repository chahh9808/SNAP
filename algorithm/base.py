import torch
import torch.nn as nn
from collections import defaultdict

from models.batch_norm import MectaNorm2d



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
                # LayerNorm lacks momentum/track_running_stats, so only toggle requires_grad
                m.requires_grad_(adapt)
                
    def print_first_bn_layer_stats(self):

        bn_layer = None
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                bn_layer = layer
                break

        if bn_layer is None:
            print("No BN layer found in the model.")
            return

        print("First BN layer statistics:")
        print("  Batch mean:", bn_layer.running_mean[0])
        print("  Batch variance:", bn_layer.running_var[0])
        print("  gamma:", bn_layer.weight)
        print("  beta:", bn_layer.bias)
    
    def print_first_ln_layer_stats(self):
        ln_layer = None
        for layer in self.model.modules():
            if isinstance(layer, nn.LayerNorm):
                ln_layer = layer
                break

        if ln_layer is None:
            print("No LayerNorm layer found in the model.")
            return

        print("First LayerNorm layer statistics:")
        if ln_layer.elementwise_affine:
            print("  gamma (weight):", ln_layer.weight)
            print("  beta (bias):", ln_layer.bias)
        else:
            print("  This LayerNorm layer does not use affine parameters.")

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
