import random
import copy
import torch
import torch.nn.functional as F
import numpy as np
import math

class FIFO():
    def __init__(self, capacity):
        self.data = [[]]
        self.capacity = capacity
        pass

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 1)

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass

class Reservoir(): # Time uniform

    def __init__(self, capacity):
        super(Reservoir, self).__init__(capacity)
        self.data = [[]]
        self.capacity = capacity
        self.counter = 0

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])


    def add_instance(self, instance):
        assert (len(instance) == 1)
        is_add = True
        self.counter+=1

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance()

        if is_add:
            for i, dim in enumerate(self.data):
                dim.append(instance[i])


    def remove_instance(self):


        m = self.get_occupancy()
        n = self.counter
        u = random.uniform(0, 1)
        if u <= m / n:
            tgt_idx = random.randrange(0, m)  # target index to remove
            for dim in self.data:
                dim.pop(tgt_idx)
        else:
            return False
        return True

num_class = 10 #CIFAR10C

class PBRS():

    def __init__(self, capacity):
        self.data = [[[]] for _ in range(num_class)] #feat, pseudo_cls, domain, cls, loss
        self.counter = [0] * num_class
        self.marker = [''] * num_class
        self.capacity = capacity
        pass
    def print_class_dist(self):

        print(self.get_occupancy_per_class())
    def print_real_class_dist(self):

        occupancy_per_class = [0] * num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] +=1
        print(occupancy_per_class)

    def get_memory(self):

        data = self.data

        tmp_data = []
        for data_per_cls in data:
            feats = data_per_cls
            tmp_data.extend(feats)
        
        tmp_data = [element for sublist in tmp_data for element in sublist]

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def update_loss(self, loss_list):
        for data_per_cls in self.data:
            feats, _, losses = data_per_cls
            for i in range(len(losses)):
                losses[i] = loss_list.pop(0)

    def add_instance(self, instance):
        assert (len(instance) == 3)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True
    
class PB():

    def __init__(self, capacity):
        self.data = [[[]] for _ in range(num_class)] #feat, pseudo_cls, domain, cls, loss
        self.counter = [0] * num_class
        self.marker = [''] * num_class
        self.capacity = capacity
        pass
    def print_class_dist(self):

        print(self.get_occupancy_per_class())
    def print_real_class_dist(self):

        occupancy_per_class = [0] * num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] +=1
        print(occupancy_per_class)

    def get_memory(self):

        data = self.data

        tmp_data = []
        for data_per_cls in data:
            feats = data_per_cls
            tmp_data.extend(feats)
        
        tmp_data = [element for sublist in tmp_data for element in sublist]

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def update_loss(self, loss_list):
        for data_per_cls in self.data:
            feats, _, losses = data_per_cls
            for i in range(len(losses)):
                losses[i] = loss_list.pop(0)

    def add_instance(self, instance):
        assert (len(instance) == 3)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:
            for dim in self.data[cls]:
                dim.pop(0)
        return True

##############################################################################

## NORMAL
class NMemory():
    def __init__(self, capacity, num_class=10,max_age_threshold=None):
        self.data = [[],[],[],[],[],[]]  # feat, entropy, confidence, age, wasserstein distance, stats 
        self.counter = 0
        self.capacity = capacity
        self.max_age_threshold = max_age_threshold
        self.aged_indices = None
    
    def reset_memory(self):
        self.data = [[],[],[],[],[],[]]
        self.counter = 0
        self.aged_indices = None

    def print_class_dist(self):
        print(self.get_occupancy_per_class())

    def get_memory(self):
        return self.data[0]

    def get_occupancy(self):
        return len(self.data[0])

    def update_entr(self, entr_list):
        if len(self.data[1])==len(entr_list):
            self.data[1] = entr_list
        else:
            assert('logit_list len doesnt match')

    def add_instance(self, instance, remove_method='RAND'):
        assert remove_method in ['RAND', 'ENTR', 'CONF', 'RS', 'WASS', 'WASS_OPP']
        assert len(instance) == 7
        self.counter += 1
        is_add = True

        # if is_add:
        #     for i, dim in enumerate(self.data):
        #         dim.append(instance[i])
        
        if is_add:
            for i, dim in enumerate(self.data):
                dim.append(instance[i])
        
        if self.get_occupancy() > self.capacity:
            is_add = self.remove_instance(remove_method=remove_method)

        # if is_add:
        #     for i, dim in enumerate(self.data):
        #         dim.append(instance[i])

    # REMOVE WAYS
    def remove_instance(self, remove_method='RAND',prediction=None):
        if self.max_age_threshold is not None:
            indexes = list(range(len(self.data)))
            random.shuffle(indexes)
            for idx in indexes:
                age = self.data[3][idx]
                if age >= self.max_age_threshold:
                    # remove_prob = sigmoid((age-self.max_age_threshold)/10-3)
                    # if remove_prob < random.uniform(0, 1):
                        for dim in self.data:
                            dim.pop(idx)
                        return True
        
        if remove_method == 'RAND':
            tgt_idx = random.randrange(0, len(self.data[0]))  # target index to remove
            for dim in self.data:
                dim.pop(tgt_idx)
            return True
        elif remove_method == 'FIFO':
            for dim in self.data:
                dim.pop(0)
            return True
        elif remove_method == 'RS':
            m = self.get_occupancy()
            n = self.counter
            u = random.uniform(0, 1)
            if u <= m / n:
                tgt_idx = random.randrange(0, m)  # target index to remove
                for dim in self.data:
                    dim.pop(tgt_idx)
            else:
                return False
            return True
        elif remove_method == 'CONF':
            confidence = self.data[2]
            min_confidence = min(confidence)
            min_confidence_index = confidence.index(min_confidence)
            for dim in self.data:
                dim.pop(min_confidence_index)
            return True
        elif remove_method == 'ENTR':
            entropy = self.data[1]
            min_entropy = min(entropy)
            min_entropy_index = entropy.index(min_entropy)
            for dim in self.data:
                dim.pop(min_entropy_index)
            return True
        elif remove_method == 'WASS': # remove low w_dist_test first
            wdist_test = self.data[4]
            min_wdist_test = min(wdist_test)
            min_wdist_test_index = wdist_test.index(min_wdist_test)
            for dim in self.data:
                dim.pop(min_wdist_test_index)
            return True
        elif remove_method == 'WASS_OPP': # remove high w_dist_test first
            wdist_test = self.data[4]
            max_wdist_test = max(wdist_test)
            max_wdist_test_index = wdist_test.index(max_wdist_test)
            for dim in self.data:
                dim.pop(max_wdist_test_index)
            return True

    # def compute_confidence(self, logits):
    #     return max(logits)

    # def compute_entropy(self, logits):
    #     temprature = 1
    #     x = logits/ temprature
    #     x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    #     return x


## ONLY FOR PB
class PBMemory():
    def __init__(self, capacity, num_class=10,max_age_threshold=None):
        self.data = [[[],[],[],[],[],[]] for _ in range(num_class)]  # feat, entropy, confidence, age, wasserstein distance, stats 
        self.counter = [0] * num_class
        self.capacity = capacity
        self.marker = [''] * num_class
        self.max_age_threshold = max_age_threshold
        self.aged_indices = None
        self.prelim_list = [] # for preliminary experiment (list of entropy and confidence values)
        
    def reset_memory(self):
        num_class = len(self.counter)
        self.data = [[[],[],[],[],[],[]] for _ in range(num_class)] 
        self.counter = [0] * num_class
        self.marker = [''] * num_class
        self.aged_indices = None

    def print_class_dist(self):
        print(self.get_occupancy_per_class())
    
    def print_age_dist(self):
        print(self.get_age_per_class())

    def get_memory(self):
        data = self.data
        tmp_data = []
        aged_indices = [] 
        current_idx = 0  

        for data_per_cls in data:
            for idx in range(len(data_per_cls[3])):
                data_per_cls[3][idx]+=1
                if data_per_cls[3][idx] > 1:
                    aged_indices.append(current_idx+idx)
            current_idx += len(data_per_cls[3])
            feats = [data_per_cls[0]]
            tmp_data.extend(feats)
        tmp_data = [element for sublist in tmp_data for element in sublist]
        self.aged_indices = aged_indices
        return tmp_data

    def get_memory_stats(self):
        data = self.data
        memory_list = [] # confidence, entropy stats, predicted class_idx

        for class_idx, data_per_cls in enumerate(data):
            for idx in range(len(data_per_cls[3])):
                memory_list.append((data_per_cls[1][idx], data_per_cls[2][idx], class_idx))

        return memory_list
    
    def get_aged_indicies(self):
        return self.aged_indices            

    def get_occupancy(self):
        return sum(len(data_per_cls[0]) for data_per_cls in self.data)

    def get_occupancy_per_class(self):
        return [len(data_per_cls[0]) for data_per_cls in self.data]
    
    def get_age_per_class(self):
        return [data_per_cls[3] for data_per_cls in self.data]

    def update_stats(self, entr_list):
        for i, data_per_cls in enumerate(self.data):
            if len(data_per_cls[1]) == len(entr_list[i]):
                data_per_cls[1] = entr_list[i]
            else:
                assert('logit_list len doesnt match')
    
    def rank_scores(self, scores):
        return [sorted(scores).index(x) for x in scores]

    def add_instance(self, instance, remove_method='RAND'):
        assert remove_method in ['RAND', 'FIFO', 'RS', 'CONF', 'ENTR', 'RSENTR', 'ENCO', 'ENRA', 'WASS', 'WASS_OPP', 'ENTR_OPP']
        assert len(instance) == 7
        cls = instance[-1]
        self.counter[cls] += 1
        is_add = True
        
        if is_add:
            for i, dim in enumerate(self.data[cls]):
                # if i==3: dim.append(0)
                # elif i==4: dim.append(instance[3])
                # else:
                dim.append(instance[i])
        
        if self.get_occupancy() > self.capacity:
            is_add = self.remove_instance(remove_method=remove_method,prediction=cls)

        # if self.get_occupancy() >= self.capacity:
        #     is_add = self.remove_instance(remove_method=remove_method,prediction=cls)
        # if is_add:
        #     for i, dim in enumerate(self.data[cls]):
        #         if i==3: dim.append(0)
        #         else: dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        return [i for i, oc in enumerate(occupancy_per_class) if oc == max_value]

    # REMOVE WAYS
    def remove_instance(self, remove_method='RAND',prediction=None):        
        if self.max_age_threshold is not None:
            indexes = list(range(len(self.data)))
            random.shuffle(indexes)
            for idx in indexes:
                cls, data_per_cls = idx, self.data[idx]
            # for cls, data_per_cls in enumerate(self.data):
                for idx, age in enumerate(data_per_cls[3]):
                    if age >= self.max_age_threshold:
                        # remove_prob = sigmoid((age-self.max_age_threshold)/10-3)
                        # if remove_prob < random.uniform(0, 1):
                            prelim_entry = [None, None, None]
                            for i, dim in enumerate(self.data[cls]):
                                if i == 1:
                                    prelim_entry[0] = dim[idx]
                                elif i == 2:
                                    prelim_entry[1] = dim[idx]
                                elif i == 3:
                                    prelim_entry[2] = dim[idx]
                                dim.pop(idx)

                            self.prelim_list.append(tuple(prelim_entry))
                            return True
        
        if remove_method == 'RAND':
            largest_indices = self.get_largest_indices()
            cls = prediction
            if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
                largest = random.choice(largest_indices)  # select only one largest class
                tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
                prelim_entry = [None, None, None]
                for i, dim in enumerate(self.data[largest]):
                    if i == 1:
                        prelim_entry[0] = dim[tgt_idx]
                    elif i == 2:
                        prelim_entry[1] = dim[tgt_idx]
                    elif i == 3:
                        prelim_entry[2] = dim[tgt_idx]
                    dim.pop(tgt_idx)
                self.prelim_list.append(tuple(prelim_entry))
            else:
                tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                prelim_entry = [None, None, None]
                for i, dim in enumerate(self.data[cls]):
                    if i == 1:
                        prelim_entry[0] = dim[tgt_idx]
                    elif i == 2:
                        prelim_entry[1] = dim[tgt_idx]
                    elif i == 3:
                        prelim_entry[2] = dim[tgt_idx]
                    dim.pop(tgt_idx)
                self.prelim_list.append(tuple(prelim_entry))
            return True
        elif remove_method == 'FIFO':
            largest_indices = self.get_largest_indices()
            cls = prediction
            if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
                largest = random.choice(largest_indices)  # select only one largest class
                prelim_entry = [None, None, None]
                for i, dim in enumerate(self.data[largest]):
                    if i == 1:
                        prelim_entry[0] = dim[0]
                    elif i == 2:
                        prelim_entry[1] = dim[0]
                    elif i == 3:
                        prelim_entry[2] = dim[0]
                    dim.pop(0)
                self.prelim_list.append(tuple(prelim_entry))
            else:
                prelim_entry = [None, None, None]
                for i, dim in enumerate(self.data[cls]):
                    if i == 1:
                        prelim_entry[0] = dim[0]
                    elif i == 2:
                        prelim_entry[1] = dim[0]
                    elif i == 3:
                        prelim_entry[2] = dim[0]
                    dim.pop(0)
                self.prelim_list.append(tuple(prelim_entry))
            return True        
        elif remove_method == 'RS':
            largest_indices = self.get_largest_indices()
            cls = prediction
            if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
                largest = random.choice(largest_indices)  # select only one largest class
                tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
                for dim in self.data[largest]:
                    self.prelim_list.append((dim[1][tgt_idx], dim[2][tgt_idx], dim[3][tgt_idx]))
                    dim.pop(tgt_idx)
            else:# replaces a randomly selected stored instance of the same class
                m_c = self.get_occupancy_per_class()[cls]
                n_c = self.counter[cls]
                u = random.uniform(0, 1)
                if u <= m_c / n_c:
                    tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                    for dim in self.data[cls]:
                        self.prelim_list.append((dim[1][tgt_idx], dim[2][tgt_idx], dim[3][tgt_idx]))
                        dim.pop(tgt_idx)
                else:
                    return False
            return True
        # MUST CONSIDER LATENCY LATER - torch.jit?
        elif remove_method == 'CONF':
            largest_indices = self.get_largest_indices()
            cls = prediction
            if cls not in largest_indices:
                min_confidence = float('inf')
                min_confidence_index = None
                min_confidence_feat_index = None
                for i in largest_indices:
                    confidence = self.data[i][2]
                    if min(confidence) < min_confidence:
                        min_confidence = min(confidence)
                        min_confidence_index = i
                        min_confidence_feat_index = confidence.index(min_confidence)
                if min_confidence_feat_index is not None:
                    for dim in self.data[min_confidence_index]:
                        #self.prelim_list.append((dim[1][min_confidence_feat_index], dim[2][min_confidence_feat_index], dim[3][min_confidence_feat_index]))
                        dim.pop(min_confidence_feat_index)
                    return True
                return False
            else:
                confidence = self.data[cls][2]
                min_confidence = min(confidence)
                min_confidence_index = confidence.index(min_confidence)
                for dim in self.data[cls]:
                    #self.prelim_list.append((dim[1][min_confidence_index], dim[2][min_confidence_index], dim[3][min_confidence_index]))
                    dim.pop(min_confidence_index)
                return True
            
        elif remove_method == 'ENTR':
            largest_indices = self.get_largest_indices()
            cls = prediction
            if cls not in largest_indices:
                min_entropy = float('inf')
                min_entropy_index = None
                min_entropy_feat_index = None
                for i in largest_indices:
                    entropy = self.data[i][1]
                    if min(entropy) < min_entropy:
                        min_entropy = min(entropy)
                        min_entropy_index = i
                        min_entropy_feat_index = entropy.index(min_entropy)
                if min_entropy_feat_index is not None:
                    prelim_entry = [None, None, None]
                    for i, dim in enumerate(self.data[min_entropy_index]):
                        if i == 1:
                            prelim_entry[0] = dim[min_entropy_feat_index]
                        elif i == 2:
                            prelim_entry[1] = dim[min_entropy_feat_index]
                        elif i == 3:
                            prelim_entry[2] = dim[min_entropy_feat_index]

                        dim.pop(min_entropy_feat_index)
                    self.prelim_list.append(tuple(prelim_entry))
                    return True
                return False
            else:
                entropy = self.data[cls][1]
                min_entropy = min(entropy)
                min_entropy_index = entropy.index(min_entropy)
                prelim_entry = [None, None, None]
                for i, dim in enumerate(self.data[cls]):
                    if i == 1:
                        prelim_entry[0] = dim[min_entropy_index]
                    if i == 2:
                        prelim_entry[1] = dim[min_entropy_index]
                    if i == 3:
                        prelim_entry[2] = dim[min_entropy_index]

                    dim.pop(min_entropy_index)
                self.prelim_list.append(tuple(prelim_entry))
                return True
                              
        elif remove_method == 'ENRA':
            largest_indices = self.get_largest_indices()
            cls = prediction
            if random.choice([True, False]):
                # Random removal
                if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
                    largest = random.choice(largest_indices)  # select only one largest class
                    tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
                    for dim in self.data[largest]:
                        #self.prelim_list.append((dim[1][tgt_idx], dim[2][tgt_idx], dim[3][tgt_idx]))
                        dim.pop(tgt_idx)
                else:
                    tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                    for dim in self.data[cls]:
                        #self.prelim_list.append((dim[1][tgt_idx], dim[2][tgt_idx], dim[3][tgt_idx]))
                        dim.pop(tgt_idx)
                return True
            else:
                if cls not in largest_indices:
                    min_entropy = float('inf')
                    min_entropy_index = None
                    min_entropy_feat_index = None
                    for i in largest_indices:
                        entropy = self.data[i][1]
                        if min(entropy) < min_entropy:
                            min_entropy = min(entropy)
                            min_entropy_index = i
                            min_entropy_feat_index = entropy.index(min_entropy)
                    if min_entropy_feat_index is not None:
                        for dim in self.data[min_entropy_index]:
                            #self.prelim_list.append((dim[1][min_entropy_feat_index], dim[2][min_entropy_feat_index], dim[3][min_entropy_feat_index]))
                            dim.pop(min_entropy_feat_index)
                        return True
                    return False
                else:
                    entropy = self.data[cls][1]
                    min_entropy = min(entropy)
                    min_entropy_index = entropy.index(min_entropy)
                    for dim in self.data[cls]:
                        #self.prelim_list.append((dim[1][min_entropy_index], dim[2][min_entropy_index], dim[3][min_entropy_index]))
                        dim.pop(min_entropy_index)
                    return True
        
        elif remove_method == 'ENCO':
            largest_indices = self.get_largest_indices()
            cls = prediction
            if cls not in largest_indices:
                min_combined_score = float('inf')
                min_combined_index = None
                min_combined_feat_index = None
                for i in largest_indices:
                    entropy = self.data[i][1]
                    confidence = self.data[i][2]
                    rank_entropy = self.rank_scores(entropy)
                    rank_confidence = self.rank_scores(confidence)
                    combined_score = [re + rc for re, rc in zip(rank_entropy, rank_confidence)]
                    if min(combined_score) < min_combined_score:
                        min_combined_score = min(combined_score)
                        min_combined_index = i
                        min_combined_feat_index = combined_score.index(min_combined_score)
                if min_combined_feat_index is not None:
                    for dim in self.data[min_combined_index]:
                        #self.prelim_list.append((dim[1][min_combined_feat_index], dim[2][min_combined_feat_index], dim[3][min_combined_feat_index]))
                        dim.pop(min_combined_feat_index)
                    return True
                return False
            else:
                entropy = self.data[cls][1]
                confidence = self.data[cls][2]
                rank_entropy = self.rank_scores(entropy)
                rank_confidence = self.rank_scores(confidence)
                combined_score = [re + rc for re, rc in zip(rank_entropy, rank_confidence)]
                min_combined_score = min(combined_score)
                min_combined_index = combined_score.index(min_combined_score)
                for dim in self.data[cls]:
                    #self.prelim_list.append((dim[1][min_combined_index], dim[2][min_combined_index], dim[3][min_combined_index]))
                    dim.pop(min_combined_index)
                return True
                
            
        elif remove_method == 'RSENTR':
            largest_indices = self.get_largest_indices()
            cls = prediction
            if cls not in largest_indices:
                min_entropy = float('inf')
                min_entropy_index = None
                min_entropy_feat_index = None
                for i in largest_indices:
                    entropy = self.data[i][1]
                    if min(entropy) < min_entropy:
                        min_entropy = min(entropy)
                        min_entropy_index = i
                        min_entropy_feat_index = entropy.index(min_entropy)
                if min_entropy_feat_index is not None:
                    m_c = self.get_occupancy_per_class()[min_entropy_index]
                    n_c = self.counter[min_entropy_index]
                    u = random.uniform(0, 1)
                    if u <= m_c / n_c:
                        for dim in self.data[min_entropy_index]:
                            #self.prelim_list.append((dim[1][min_entropy_feat_index], dim[2][min_entropy_feat_index], dim[3][min_entropy_feat_index]))
                            dim.pop(min_entropy_feat_index)
                        return True
                    return False
            else:
                m_c = self.get_occupancy_per_class()[cls]
                n_c = self.counter[cls]
                u = random.uniform(0, 1)
                if u <= m_c / n_c:
                    entropy = self.data[cls][1]
                    min_entropy = min(entropy)
                    min_entropy_feat_index = entropy.index(min_entropy)
                    for dim in self.data[cls]:
                        #self.prelim_list.append((dim[1][min_entropy_feat_index], dim[2][min_entropy_feat_index], dim[3][min_entropy_feat_index]))
                        dim.pop(min_entropy_feat_index)
                    return True
                return False
            
        elif remove_method == 'WASS': # remove low w_dist first
            largest_indices = self.get_largest_indices()
            cls = prediction
            if cls not in largest_indices:
                min_w_dist = float('inf')
                min_w_dist_index = None
                min_w_dist_feat_index = None
                for i in largest_indices:
                    w_dist = self.data[i][4]
                    if min(w_dist) < min_w_dist:
                        min_w_dist = min(w_dist)
                        min_w_dist_index = i
                        min_w_dist_feat_index = w_dist.index(min_w_dist)
                if min_w_dist_feat_index is not None:
                    prelim_entry = [None, None, None]
                    for i, dim in enumerate(self.data[min_w_dist_index]):
                        if i == 1:
                            prelim_entry[0] = dim[min_w_dist_feat_index]
                        elif i == 2:
                            prelim_entry[1] = dim[min_w_dist_feat_index]
                        elif i == 3:
                            prelim_entry[2] = dim[min_w_dist_feat_index]

                        dim.pop(min_w_dist_feat_index)
                    self.prelim_list.append(tuple(prelim_entry))
                    return True
                return False
            else:
                w_dist = self.data[cls][4]
                min_w_dist = min(w_dist)
                min_w_dist_index = w_dist.index(min_w_dist)
                prelim_entry = [None, None, None]
                for i, dim in enumerate(self.data[cls]):
                    if i == 1:
                        prelim_entry[0] = dim[min_w_dist_index]
                    if i == 2:
                        prelim_entry[1] = dim[min_w_dist_index]
                    if i == 3:
                        prelim_entry[2] = dim[min_w_dist_index]

                    dim.pop(min_w_dist_index)
                self.prelim_list.append(tuple(prelim_entry))
                return True
            
        elif remove_method == 'WASS_OPP': # remove high w_dist first
            largest_indices = self.get_largest_indices()
            cls = prediction
            if cls not in largest_indices:
                max_w_dist  = float('-inf')
                max_w_dist_index  = None
                max_w_dist_feat_index = None
                for i in largest_indices:
                    w_dist = self.data[i][4]
                    if max(w_dist) > max_w_dist:
                        max_w_dist = max(w_dist)
                        max_w_dist_index = i
                        max_w_dist_feat_index = w_dist.index(max_w_dist)
                if max_w_dist_feat_index is not None:
                    prelim_entry = [None, None, None]
                    for i, dim in enumerate(self.data[max_w_dist_index]):
                        if i == 1:
                            prelim_entry[0] = dim[max_w_dist_feat_index]
                        elif i == 2:
                            prelim_entry[1] = dim[max_w_dist_feat_index]
                        elif i == 3:
                            prelim_entry[2] = dim[max_w_dist_feat_index]

                        dim.pop(max_w_dist_feat_index)
                    self.prelim_list.append(tuple(prelim_entry))
                    return True            
                return False
            
            else:
                w_dist = self.data[cls][4]
                max_w_dist = max(w_dist)
                max_w_dist_index = w_dist.index(max_w_dist)
                prelim_entry = [None, None, None]
                for i, dim in enumerate(self.data[cls]):
                    if i == 1:
                        prelim_entry[0] = dim[max_w_dist_index]
                    if i == 2:
                        prelim_entry[1] = dim[max_w_dist_index]
                    if i == 3:
                        prelim_entry[2] = dim[max_w_dist_index]

                    dim.pop(max_w_dist_index)
                self.prelim_list.append(tuple(prelim_entry))
                return True
            
        elif remove_method == 'ENTR_OPP':
            largest_indices = self.get_largest_indices()
            cls = prediction
            if cls not in largest_indices:
                max_entropy = float('-inf')
                max_entropy_index = None
                max_entropy_feat_index = None
                for i in largest_indices:
                    entropy = self.data[i][1]
                    if max(entropy) > max_entropy:
                        max_entropy = max(entropy)
                        max_entropy_index = i
                        max_entropy_feat_index = entropy.index(max_entropy)
                if max_entropy_feat_index is not None:
                    prelim_entry = [None, None, None]
                    for i, dim in enumerate(self.data[max_entropy_index]):
                        if i == 1:
                            prelim_entry[0] = dim[max_entropy_feat_index]
                        elif i == 2:
                            prelim_entry[1] = dim[max_entropy_feat_index]
                        elif i == 3:
                            prelim_entry[2] = dim[max_entropy_feat_index]

                        dim.pop(max_entropy_feat_index)
                    self.prelim_list.append(tuple(prelim_entry))
                    return True
                return False
            else:
                entropy = self.data[cls][1]
                max_entropy = max(entropy)
                max_entropy_index = entropy.index(max_entropy)
                prelim_entry = [None, None, None]
                for i, dim in enumerate(self.data[cls]):
                    if i == 1:
                        prelim_entry[0] = dim[max_entropy_index]
                    if i == 2:
                        prelim_entry[1] = dim[max_entropy_index]
                    if i == 3:
                        prelim_entry[2] = dim[max_entropy_index]

                    dim.pop(max_entropy_index)
                self.prelim_list.append(tuple(prelim_entry))
                return True


    # def compute_confidence(self, logits):
    #     return max(logits)

    # def compute_entropy(self, logits):
    #     temprature = 1
    #     x = logits/ temprature
    #     x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    #     return x
        # total = sum(logits)
        # probabilities = [logit / total for logit in logits]
        # entropy = -sum(p * math.log2(p) for p in probabilities if p != 0)
        # return entropy
            
def sigmoid(x):
        return 1 / (1 + math.exp(-x))