import torch

from utils.iobmn import SparseAdaptationAwareBatchNorm2d,SparseAdaptationAwareLayerNorm
from models.batch_norm import MectaNorm2d


def wasserstein_distance(mu_s, sigma2_s, mu_t, sigma2_t):
    # calculate the squared Euclidean distance between means
    mean_distance = torch.sum((mu_s - mu_t) ** 2)

    # calculate the simplified trace term assuming diagonal covariance matrices
    trace_term = torch.sum(sigma2_s + sigma2_t - 2 * torch.sqrt(sigma2_s * sigma2_t))

    # wasserstein distance squared
    wasserstein_distance_sq = mean_distance + trace_term
    wasserstein_distance_sq = torch.sqrt(wasserstein_distance_sq)

    # square root to get wasserstein distance
    wasserstein_distance = torch.sqrt(wasserstein_distance_sq)

    return wasserstein_distance.item()

# gets first layer BN stats
def bn_iobmn_get_bn_stats(self, compare_with_test=False,layer_t=0):
    count = 0
    for module in self.model.modules():
        if isinstance(module, SparseAdaptationAwareBatchNorm2d) or isinstance(module,SparseAdaptationAwareLayerNorm) or isinstance(module, MectaNorm2d):
            if count == layer_t:
                if compare_with_test:
                    return module.mu_batch, module.sigma2_batch, module.mu_test, module.sigma2_test
                else:
                    return module.mu_batch, module.sigma2_batch, module._bn.training_mean, module._bn.training_var
            count += 1
    
    print("ERROR: SparseAdaptationAwareBatchNorm2d not found")
    return None, None, None, None

# returns first N BN layer stats
def bn_iobmn_get_bn_stats_N(self, N, compare_with_test=False):
    # N=None means iterate through all BN layers
    stats = []
    count = 0
    for module in self.model.modules():
        if isinstance(module, SparseAdaptationAwareBatchNorm2d):
            if compare_with_test:
                stats.append((module.mu_batch, module.sigma2_batch, module.mu_test, module.sigma2_test))
            else:
                stats.append((module.mu_batch, module.sigma2_batch, module._bn.training_mean, module._bn.training_var))
            count += 1
            if N is not None and count == N:
                break
    
    if N is not None and len(stats) != N:
        print(f"ERROR: fewer than {N} SparseAdaptationAwareBatchNorm2d were found")
        return None
    
    return stats

def bn_retrieve_bn_stats(self, outputs, isadapt,layer_t=0):
    mu_batch, sigma2_batch, mu_test, sigma2_test = self.iobmn_get_bn_stats(compare_with_test=True,layer_t=layer_t)

    mu_comp = mu_test
    sigma2_comp = sigma2_test
    if isadapt: #on-off?
        self.mu_centr = mu_test
        self.sigma2_centr = sigma2_test
    else:
        if self.mu_centr is not None and self.sigma2_centr is not None:
            mu_comp = self.mu_centr
            sigma2_comp = self.sigma2_centr
            
    stats_list = []
    # wasserstein distance
    wasserstein_distances = []
    entropies = softmax_entropy(outputs)
    # iterate over each image
    for i in range(outputs.size(0)):
        # wasserstein distance
        w_dist = wasserstein_distance(mu_batch[i], sigma2_batch[i], mu_comp, sigma2_comp)
        # store entropy and distances as a tuple, and append to global list
        per_image_stat = (entropies[i].item(), w_dist, None)
        self.bn_analysis.append(per_image_stat)
        wasserstein_distances.append(w_dist)
        stats_list.append([mu_batch[i],sigma2_batch[i]])

    return wasserstein_distances, stats_list, mu_test, sigma2_test

# calculate WDIST using first N layer BN stats, normalize each layer WDIST using running average of WDIST stats so that each layer gets fair weight,
# then sums all layers to get one WDIST per image
# return list of WDIST for each image
# DOES NOT HAVE LOGIC TO HANDLE RECALCULATION
def bn_retrieve_bn_stats_N(self, outputs, N, isadapt):
    stats_test = self.iobmn_get_bn_stats_N(N, compare_with_test=True)

    entropies = softmax_entropy(outputs)

    wasserstein_distances_test = [[] for _ in range(N)]

    # iterate over each image
    for i in range(outputs.size(0)):
        # iterate over each BN layer
        for layer_idx, stat_layer in enumerate(stats_test):
            mu_batch, sigma2_batch, mu_test, sigma2_test = stat_layer
            
            mu_comp = mu_test
            sigma2_comp = sigma2_test
            
            #TO-DO
            #SHOULD MAKE mu_centr and sigma_centr to list form. 
            # freezing
            if isadapt:
                self.mu_centr[0] = mu_test  
                self.sigma2_centr = sigma2_test
            else:
                if self.mu_centr is not None and self.sigma2_centr is not None:
                    mu_comp = self.mu_centr
                    sigma2_comp = self.sigma2_centr
            

            w_dist = wasserstein_distance(mu_comp, sigma2_comp, mu_batch[i], sigma2_batch[i])
            wasserstein_distances_test[layer_idx].append(w_dist)
        
    wasserstein_means_test = [torch.mean(torch.tensor(distances)) for distances in wasserstein_distances_test] # mean of WDIST for each layer 
    wasserstein_vars_test = [torch.var(torch.tensor(distances)) for distances in wasserstein_distances_test] # variance of WDIST for each layer

    # update running estimate of wasserstein_means and wasserstein_vars
    if self.wasserstein_means_test is None:
        self.wasserstein_means_test = wasserstein_means_test
        self.wasserstein_vars_test = wasserstein_vars_test

    else:
        # use momentum update
        self.wasserstein_means_test = [self.norm_beta * batch_mean + (1 - self.norm_beta) * running_mean for batch_mean, running_mean in zip(wasserstein_means_test, self.wasserstein_means_test)]
        self.wasserstein_vars_test = [self.norm_beta * batch_var + (1 - self.norm_beta) * running_var for batch_var, running_var in zip(wasserstein_vars_test, self.wasserstein_vars_test)]

    normalized_wasserstein_distances_test = []

    # normalize the per layer WDISTs and sum them to get per image WDIST
    for i in range(outputs.size(0)):
        w_dist_sum_test = 0
        
        for layer_idx in range(N):
            w_dist = wasserstein_distances_test[layer_idx][i]

            mean = self.wasserstein_means_test[layer_idx]
            var = self.wasserstein_vars_test[layer_idx]
            w_dist_normalized = torch.abs((w_dist - mean) / torch.sqrt(var + 1e-8))
            
            w_dist_sum_test += w_dist_normalized.item()

        # store entropy and WDIST as a tuple, and append to global list
        per_image_stat = (entropies[i].item(), w_dist_sum_test, None)
        self.bn_analysis.append(per_image_stat)

        # append stats to list to be returned
        normalized_wasserstein_distances_test.append(w_dist_sum_test)

    return normalized_wasserstein_distances_test

def bn_recalc_bn_stats(self, memtype,compare_with_test=True):
    _, _, mu_test, sigma2_test = self.iobmn_get_bn_stats(compare_with_test,self.layer_t)

    # update frozen moving avg
    self.mu_centr = mu_test
    self.sigma2_centr = sigma2_test

    tot_idx = 0
    if self.memory is not None:
        if memtype == 'pb':
            for data_per_cls in self.memory.data:
                for idx in range(len(data_per_cls[3])):
                    w_dist = wasserstein_distance(data_per_cls[5][idx][0], data_per_cls[5][idx][1], mu_test, sigma2_test)
                    data_per_cls[4][idx] = w_dist
                    tot_idx=tot_idx+1
        else:
            for idx in range(len(self.memory.data[3])):
                w_dist = wasserstein_distance(self.memory.data[5][idx][0], self.memory.data[5][idx][1], mu_test, sigma2_test)
                self.memory.data[4][idx] = w_dist
                tot_idx=tot_idx+1
                
    self.cnt = self.cnt+1
    return tot_idx

def bn_check_bn_divergence(self,memtype, mu, sigma2):
    if self.mu_centr is not None:
        # calculate difference between frozen moving avg and current moving avg
        diff = wasserstein_distance(self.mu_centr, self.sigma2_centr, mu, sigma2)
        self.wass_dist.append(diff)
        if diff > 0.1:
            self.recalc_bn_stats(memtype,compare_with_test=True)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x
