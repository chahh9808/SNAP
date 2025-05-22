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

def prepare_imagenet_test_data_dirichlet_skew(corruption, level, batch_size,
                                              workers=1, seed=None, alpha=0.1,
                                              num_classes=1000):
    """
    Load ImageNet test data and apply Dirichlet(alpha)-based label skew.

    :param corruption: Corruption type ('original' or corruption name from IN-C).
    :param level: Corruption severity level (for IN-C).
    :param batch_size: Batch size for DataLoader.
    :param workers: DataLoader worker threads.
    :param seed: Random seed.
    :param alpha: Dirichlet alpha for label skewing (smaller = more skewed).
    :param num_classes: Limit dataset to N classes (default = 1000).
    :return: (skewed_dataset, DataLoader)
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    normalize = trns.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

    if corruption == 'original':
        te_transforms = trns.Compose([
            trns.Resize(256), trns.CenterCrop(224), trns.ToTensor(), normalize
        ])
        print('Test on the original ImageNet val set')
        val_root = os.path.join(DATA_PATHS['IN'], 'val')
        test_set = ImageFolder(val_root, te_transforms)

    elif corruption in IN_C_corruptions:
        te_transforms_imageC = trns.Compose([
            trns.CenterCrop(224), trns.ToTensor(), normalize
        ])
        print(f'Test on ImageNet-C: {corruption}, level {level}')
        val_root = os.path.join(DATA_PATHS['IN-C'], corruption, str(level))
        test_set = ImageFolder(val_root, te_transforms_imageC)

    else:
        raise Exception(f'Corruption "{corruption}" not found!')

    # Filter classes if num_classes < 1000
    if num_classes is not None and num_classes < 1000:
        labels = np.array(test_set.targets)
        idxs = np.nonzero(labels < num_classes)[0]
        test_set = Subset(test_set, indices=idxs)
        labels = labels[idxs]  # update labels accordingly
    else:
        labels = np.array(test_set.targets)

    # Dirichlet label skew
    unique_classes = np.unique(labels)
    class_indices = [np.where(labels == y)[0] for y in unique_classes]

    proportions = rng.dirichlet([alpha] * len(unique_classes))
    total_samples = len(labels)
    samples_per_class = (proportions * total_samples).astype(int)

    # Fix rounding errors
    while samples_per_class.sum() < total_samples:
        samples_per_class[rng.randint(0, len(unique_classes))] += 1
    while samples_per_class.sum() > total_samples:
        samples_per_class[rng.randint(0, len(unique_classes))] -= 1

    selected_indices = []
    for c, cls_idxs in enumerate(class_indices):
        rng.shuffle(cls_idxs)
        selected_indices.extend(cls_idxs[:samples_per_class[c]])

    rng.shuffle(selected_indices)
    skewed_subset = Subset(test_set, selected_indices)

    loader = DataLoader(skewed_subset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)

    return skewed_subset, loader


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

def prepare_cifar10_test_data_dirichlet_skew(corruption, level, batch_size,
                                             workers=1, seed=None, alpha=0.1):
    """
    Load CIFAR-10 test data and apply Dirichlet(alpha)-based label skew globally.

    :param corruption: Corruption type ('original' or other supported corruptions).
    :param level: Corruption level for corrupted data.
    :param batch_size: Batch size for the DataLoader.
    :param workers: Number of workers for the DataLoader.
    :param seed: Random seed for reproducibility.
    :param alpha: Dirichlet alpha value to skew label distribution (smaller = more skewed).
    :return: (test_set, DataLoader)
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

    labels = np.array([label for _, label in test_set])
    num_classes = np.max(labels) + 1
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]

    # Sample class proportions using Dirichlet distribution
    proportions = rng.dirichlet([alpha] * num_classes)
    total_samples = len(labels)
    samples_per_class = (proportions * total_samples).astype(int)

    # Fix rounding issues
    while samples_per_class.sum() < total_samples:
        samples_per_class[rng.randint(0, num_classes)] += 1
    while samples_per_class.sum() > total_samples:
        samples_per_class[rng.randint(0, num_classes)] -= 1

    selected_indices = []
    for c in range(num_classes):
        rng.shuffle(class_indices[c])
        selected_indices.extend(class_indices[c][:samples_per_class[c]])

    rng.shuffle(selected_indices)
    skewed_subset = Subset(test_set, selected_indices)

    loader = DataLoader(skewed_subset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    return skewed_subset, loader


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


def prepare_cifar100_test_data_dirichlet_skew(corruption, level, batch_size,
                                             workers=1, seed=None, alpha=0.01):
    """
    Load CIFAR-100 test data and apply Dirichlet(alpha)-based label skew globally.

    :param corruption: Corruption type ('original' or other supported corruptions).
    :param level: Corruption level for corrupted data.
    :param batch_size: Batch size for the DataLoader.
    :param workers: Number of workers for the DataLoader.
    :param seed: Random seed for reproducibility.
    :param alpha: Dirichlet alpha value to skew label distribution (smaller = more skewed).
    :return: (test_set, DataLoader)
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    normalize = trns.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trans = trns.Compose([trns.ToTensor(), normalize])

    if corruption == 'original':
        test_set = CIFAR100(DATA_PATHS['Cifar100'], train=False, transform=trans, download=True)
    elif corruption in CIFAR10_CORRUPTIONS:
        x_test, y_test = load_cifar100c(10000, level, DATA_PATHS['Cifar100'], True, [corruption])
        test_set = LabeledDataset(x_test, y_test, transform=None)
    else:
        raise RuntimeError(f"Not supported corruption: {corruption}")

    labels = np.array([label for _, label in test_set])
    num_classes = np.max(labels) + 1
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]

    # Sample class proportions using Dirichlet distribution
    proportions = rng.dirichlet([alpha] * num_classes)
    total_samples = len(labels)
    samples_per_class = (proportions * total_samples).astype(int)

    # Fix rounding issues
    while samples_per_class.sum() < total_samples:
        samples_per_class[rng.randint(0, num_classes)] += 1
    while samples_per_class.sum() > total_samples:
        samples_per_class[rng.randint(0, num_classes)] -= 1

    selected_indices = []
    for c in range(num_classes):
        rng.shuffle(class_indices[c])
        selected_indices.extend(class_indices[c][:samples_per_class[c]])

    rng.shuffle(selected_indices)
    skewed_subset = Subset(test_set, selected_indices)

    loader = DataLoader(skewed_subset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    return skewed_subset, loader