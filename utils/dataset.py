import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from torchvision import transforms as trns
from torchvision.datasets import ImageFolder
from .robustbench_data import load_cifar10c, load_cifar100c, load_cifar10c_bybatch,load_cifar100c_bybatch, BenchmarkDataset, DownloadError
from utils.config import DATA_PATHS, data_root
from torchvision.datasets import CIFAR10, CIFAR100

from .zenodo_download import DownloadError, zenodo_download
from .robustbench_loaders import CustomImageFolder

from utils.cli_utils import AverageMeter, ProgressMeter, accuracy
from utils.config import DATA_PATHS
from pathlib import Path

from typing import Callable, Dict, Optional, Sequence, Set, Tuple

from collections import defaultdict


IN_C_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

# NOTE this is more than those in robustbench but included in Hendryc's dataset.
CIFAR10_CORRUPTIONS = ('saturate', 'glass_blur', 'fog', 'brightness', 'snow', 'contrast',
                       'defocus_blur', 'zoom_blur', 'jpeg_compression', 'elastic_transform',
                       'spatter', 'frost', 'gaussian_blur', 'impulse_noise', 'gaussian_noise',
                       'motion_blur', 'speckle_noise', 'pixelate', 'shot_noise')

CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform")

CORRUPTIONS_3DCC = ('near_focus', 'far_focus', 'bit_error', 'color_quant',
                    'flash', 'fog_3d', 'h265_abr', 'h265_crf', 'iso_noise',
                    'low_light', 'xy_motion_blur', 'z_motion_blur')

ZENODO_CORRUPTIONS_LINKS: Dict[BenchmarkDataset, Tuple[str, Set[str]]] = {
    BenchmarkDataset.cifar_10: ("2535967", {"CIFAR-10-C.tar"}),
    BenchmarkDataset.cifar_100: ("3555552", {"CIFAR-100-C.tar"})
}

CORRUPTIONS_DIR_NAMES: Dict[BenchmarkDataset, str] = {
    BenchmarkDataset.cifar_10: "CIFAR-10-C",
    BenchmarkDataset.cifar_100: "CIFAR-100-C",
    BenchmarkDataset.imagenet: "ImageNet-C",
    BenchmarkDataset.imagenet_3d: "ImageNet-3DCC"
}

class LabeledDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        super(LabeledDataset, self).__init__()
        assert data.size(0) == targets.size(0)
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.targets)


# //////// Prepare data loaders //////////
def prepare_imagenet_test_data(corruption, level, batch_size,
                               subset_size=None, workers=1, seed=None,
                               num_classes=1000):

    rng = np.random.RandomState(seed) if seed is not None else np.random
    normalize = trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if corruption == 'original':
        te_transforms = trns.Compose([trns.Resize(256), trns.CenterCrop(224), trns.ToTensor(),
                                      normalize])
        print('Test on the original test set')
        val_root = os.path.join(DATA_PATHS['IN'], 'val')
        test_set = ImageFolder(val_root, te_transforms)
    elif corruption in IN_C_corruptions:
        te_transforms_imageC = trns.Compose([trns.CenterCrop(224),
                                             trns.ToTensor(), normalize
                                             ])
        print('Test on %s level %d' % (corruption, level))
        val_root = os.path.join(DATA_PATHS['IN-C'], corruption, str(level))
        test_set = ImageFolder(val_root, te_transforms_imageC)
    else:
        raise Exception(f'Corruption {corruption} not found!')

    if num_classes is not None:
        idxs = np.nonzero(np.array(test_set.targets) < num_classes)[0]
        test_set = Subset(test_set, indices=idxs)

    if subset_size is not None:
        idxs = np.arange(len(test_set))
        idxs = rng.permutation(idxs)
        idxs = idxs[:subset_size]
        test_set = Subset(test_set, idxs)

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                        num_workers=workers, pin_memory=True)
    return test_set, loader

def prepare_imagenet_test_data_non_iid(corruption, level, batch_size,
                                       subset_size=None, workers=1, seed=None,
                                       num_classes=1000,skew_ratio=0.5):
    """
    Prepare non-iid test data for ImageNet based on label skew using skew_ratio.
    """
    print('non-iid')
    rng = np.random.RandomState(seed) if seed is not None else np.random
    normalize = trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if corruption == 'original':
        te_transforms = trns.Compose([trns.Resize(256), trns.CenterCrop(224), trns.ToTensor(),
                                      normalize])
        print('Test on the original test set')
        val_root = os.path.join(DATA_PATHS['IN'], 'val')
        test_set = ImageFolder(val_root, te_transforms)
    elif corruption in IN_C_corruptions:
        te_transforms_imageC = trns.Compose([trns.CenterCrop(224),
                                             trns.ToTensor(), normalize])
        print('Test on %s level %d' % (corruption, level))
        val_root = os.path.join(DATA_PATHS['IN-C'], corruption, str(level))
        test_set = ImageFolder(val_root, te_transforms_imageC)
    else:
        raise Exception(f'Corruption {corruption} not found!')

    if num_classes is not None:
        idxs = np.nonzero(np.array(test_set.targets) < num_classes)[0]
        test_set = Subset(test_set, indices=idxs)

    if subset_size is not None:
        # Create a label distribution based on skew_ratio
        targets = np.array(test_set.dataset.targets)
        label_to_idxs = defaultdict(list)
        for idx, label in enumerate(targets):
            label_to_idxs[label].append(idx)

        # Adjust the proportion of data per label
        all_idxs = []
        for label, idxs in label_to_idxs.items():
            rng.shuffle(idxs)  # Shuffle indices for each label
            keep_count = int(len(idxs) * (skew_ratio if label == 0 else (1 - skew_ratio)))
            all_idxs.extend(idxs[:keep_count])

        all_idxs = rng.permutation(all_idxs)  # Shuffle final indices
        test_set = Subset(test_set, indices=all_idxs)

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    return test_set, loader

def prepare_imagenet_test_data_bybatch(corruption, level, batch_size,
                              subset_size=None, workers=1, seed=None,idx=None,datahelper=None):
    assert datahelper is not None
    # rng = np.random.RandomState(seed) if seed is not None else np.random
    normalize = trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # if idx==None or 10000<(batch_size*(idx+1)):
    #     raise RuntimeError("idx error")

    # normalize = trns.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # trans = trns.Compose([trns.ToTensor(), normalize])
    if corruption == 'original':
        te_transforms = trns.Compose([trns.Resize(256), trns.CenterCrop(224), trns.ToTensor(),
                                      normalize])
        print('Test on the original test set')
        val_root = os.path.join(DATA_PATHS['IN'], 'val')
        test_set = ImageFolder(val_root, te_transforms)
    elif corruption in IN_C_corruptions:
        te_transforms_imageC = trns.Compose([trns.CenterCrop(224),
                                             trns.ToTensor(), normalize
                                             ])
        print('Test on %s level %d' % (corruption, level))
        val_root = os.path.join(DATA_PATHS['IN-C'], corruption, str(level))
        test_set = ImageFolder(val_root, te_transforms_imageC)
        x_test, y_test = load_imagenetc_bybatch(batch_size, idx, datahelper)
        # x_test, y_test = x_test.to(device), y_test.to(device)  # NOTE this will cause CUDA init error
        # Temporally fix: normalize->None
        test_set = LabeledDataset(x_test, y_test, transform=None)
    else:
        raise RuntimeError(f"Not supported corruption: {corruption}")

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, #True -> False
                        num_workers=workers, pin_memory=True)
    return test_set, loader


def prepare_cifar10_test_data(corruption, level, batch_size,
                              subset_size=None, workers=1, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random

    normalize = trns.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trans = trns.Compose([trns.ToTensor(), normalize])
    if corruption == 'original':
        test_set = CIFAR10(DATA_PATHS['Cifar10'], train=False, transform=trans, download=True)
    elif corruption in CIFAR10_CORRUPTIONS:
        x_test, y_test = load_cifar10c(10000, level, DATA_PATHS['Cifar10'], True, [corruption])
        # x_test, y_test = x_test.to(device), y_test.to(device)  # NOTE this will cause CUDA init error
        # Temporally fix: normalize->None
        test_set = LabeledDataset(x_test, y_test, transform=None)
    else:
        raise RuntimeError(f"Not supported corruption: {corruption}")
    if subset_size is not None:
        idxs = np.arange(len(test_set))
        idxs = rng.permutation(idxs)
        idxs = idxs[:subset_size]
        test_set = Subset(test_set, idxs)

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, #True -> False
                        num_workers=workers, pin_memory=True)
    return test_set, loader

def prepare_cifar10_test_data_non_iid_skew(corruption, level, batch_size,
                                           subset_size=None, workers=1, seed=None, skew_ratio=1.0):
    """
    Load CIFAR-10 test data in a non-iid manner with a uniform skew ratio.

    :param corruption: Corruption type ('original' or other supported corruptions).
    :param level: Corruption level for corrupted data.
    :param batch_size: Batch size for the DataLoader.
    :param subset_size: Size of the subset of the dataset (optional).
    :param workers: Number of workers for the DataLoader.
    :param seed: Random seed for reproducibility.
    :param skew_ratio: Ratio to uniformly reduce data for each label (e.g., 0.5 keeps 50% per label).
    :return: (test_set, DataLoader) where test_set is the dataset and DataLoader is the DataLoader for it.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    normalize = trns.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trans = trns.Compose([trns.ToTensor(), normalize])
    if corruption == 'original':
        test_set = CIFAR10(DATA_PATHS['Cifar10'], train=False, transform=trans, download=True)
    elif corruption in CIFAR10_CORRUPTIONS:
        x_test, y_test = load_cifar10c(10000, level, DATA_PATHS['Cifar10'], True, [corruption])
        test_set = LabeledDataset(x_test, y_test, transform=None)
    else:
        raise RuntimeError(f"Not supported corruption: {corruption}")

    # Apply skew based on skew_ratio
    if skew_ratio < 1.0:
        label_indices = {label: [] for label in range(10)}  # Assuming 10 labels in CIFAR-10
        for idx, (_, label) in enumerate(test_set):
            label_indices[label].append(idx)

        selected_indices = []
        for label, indices in label_indices.items():
            rng.shuffle(indices)
            num_samples = int(skew_ratio * len(indices))
            selected_indices.extend(indices[:num_samples])

        rng.shuffle(selected_indices)  # Shuffle the selected indices for randomization
        test_set = Subset(test_set, selected_indices)

    # If subset_size is specified, further reduce the dataset
    if subset_size is not None:
        idxs = np.arange(len(test_set))
        idxs = rng.permutation(idxs)
        idxs = idxs[:subset_size]
        test_set = Subset(test_set, idxs)

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,  # Shuffle is False for deterministic splits
                        num_workers=workers, pin_memory=True)
    return test_set, loader

def prepare_cifar10_test_data_bybatch(corruption, level, batch_size,
                              subset_size=None, workers=1, seed=None,idx=None,datahelper=None):
    assert datahelper is not None
    if idx==None or 10000<(batch_size*(idx+1)):
        raise RuntimeError("idx error")

    # normalize = trns.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # trans = trns.Compose([trns.ToTensor(), normalize])
    if corruption == 'original':
        test_set = CIFAR10(DATA_PATHS['Cifar10'], train=False, transform=None, download=True)
    elif corruption in CIFAR10_CORRUPTIONS:
        x_test, y_test = load_cifar10c_bybatch(batch_size, idx, datahelper)
        # x_test, y_test = x_test.to(device), y_test.to(device)  # NOTE this will cause CUDA init error
        # Temporally fix: normalize->None
        test_set = LabeledDataset(x_test, y_test, transform=None)
    else:
        raise RuntimeError(f"Not supported corruption: {corruption}")

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, #True -> False
                        num_workers=workers, pin_memory=True)
    return test_set, loader

def prepare_cifar100_test_data_bybatch(corruption, level, batch_size,
                              subset_size=None, workers=1, seed=None,idx=None,datahelper=None):
    assert datahelper is not None
    if idx==None or 10000<(batch_size*(idx+1)):
        raise RuntimeError("idx error")

    # normalize = trns.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # trans = trns.Compose([trns.ToTensor(), normalize])
    if corruption == 'original':
        test_set = CIFAR10(DATA_PATHS['Cifar100'], train=False, transform=None, download=True)
    elif corruption in CIFAR10_CORRUPTIONS:
        x_test, y_test = load_cifar100c_bybatch(batch_size, idx, datahelper)
        # x_test, y_test = x_test.to(device), y_test.to(device)  # NOTE this will cause CUDA init error
        # Temporally fix: normalize->None
        test_set = LabeledDataset(x_test, y_test, transform=None)
    else:
        raise RuntimeError(f"Not supported corruption: {corruption}")

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, #True -> False
                        num_workers=workers, pin_memory=True)
    return test_set, loader


def prepare_cifar100_test_data(corruption, level, batch_size,
                              subset_size=None, workers=1, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random

    normalize = trns.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trans = trns.Compose([trns.ToTensor(), normalize])
    if corruption == 'original':
        test_set = CIFAR100(DATA_PATHS['Cifar100'], train=False, transform=trans, download=True)
    elif corruption in CIFAR10_CORRUPTIONS:
        x_test, y_test = load_cifar100c(10_000, level, DATA_PATHS['Cifar100'], True, [corruption])
        # x_test, y_test = x_test.to(device), y_test.to(device)  # NOTE this will cause CUDA init error
        test_set = LabeledDataset(x_test, y_test, transform=None)
    else:
        raise RuntimeError(f"Not supported corruption: {corruption}")
    if subset_size is not None:
        idxs = np.arange(len(test_set))
        idxs = rng.permutation(idxs)
        idxs = idxs[:subset_size]
        test_set = Subset(test_set, idxs)

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    return test_set, loader
