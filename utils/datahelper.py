import os
import numpy as np
from utils.config import DATA_PATHS
from enum import Enum

from .zenodo_download import zenodo_download
import torchvision.transforms as transforms

from utils.config import DATA_PATHS
from pathlib import Path

from typing import Dict, Sequence, Set, Tuple

class BenchmarkDataset(Enum):
    cifar_10 = 'cifar10'
    cifar_100 = 'cifar100'
    imagenet = 'imagenet'
    imagenet_3d = 'imagenet_3d'


class DownloadError(Exception):
    pass

PREPROCESSINGS = {
    'Res256Crop224':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    'Crop288':
    transforms.Compose([transforms.CenterCrop(288),
                        transforms.ToTensor()]),
    None:
    transforms.Compose([transforms.ToTensor()]),
}

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

class DataHelper:
    def __init__(self, data, corruption, level, shuffle):
        if data == 'cifar10':
            x_test, y_test = load_np_corruptions_cifar10_forbybatch(level,[corruption],shuffle)
        elif data == 'cifar100':
            x_test, y_test = load_np_corruptions_cifar100_forbybatch(level,[corruption],shuffle)
        # Other datasets will add later
        self.x_test=x_test
        self.y_test=y_test
    def get_np(self):
        return self.x_test, self.y_test
    
def load_np_corruptions_cifar10_forbybatch(
        severity: int,
        corruptions: Sequence[str] = CORRUPTIONS,
        shuffle: bool = False):
    assert 1 <= severity <= 5
    n_total_cifar = 10000
    n_examples = n_total_cifar
    dataset = BenchmarkDataset.cifar_10
    data_dir = DATA_PATHS['Cifar10']

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = Path(data_dir)
    data_root_dir = data_dir / CORRUPTIONS_DIR_NAMES[dataset]

    if not data_root_dir.exists():
        zenodo_download(*ZENODO_CORRUPTIONS_LINKS[dataset], save_dir=data_dir)

    # Download labels
    labels_path = data_root_dir / 'labels.npy'
    if not os.path.isfile(labels_path):
        raise DownloadError("Labels are missing, try to re-download them.")
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + '.npy')
        if not corruption_file_path.is_file():
            raise DownloadError(
                f"{corruption} file is missing, try to re-download it.")

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity *
                            n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])
    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]
    return x_test, y_test

def load_np_corruptions_cifar100_forbybatch(
        severity: int,
        corruptions: Sequence[str] = CORRUPTIONS,
        shuffle: bool = False):
    assert 1 <= severity <= 5
    n_total_cifar = 10000
    n_examples = n_total_cifar
    dataset = BenchmarkDataset.cifar_100
    data_dir = DATA_PATHS['Cifar100']

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = Path(data_dir)
    data_root_dir = data_dir / CORRUPTIONS_DIR_NAMES[dataset]

    if not data_root_dir.exists():
        zenodo_download(*ZENODO_CORRUPTIONS_LINKS[dataset], save_dir=data_dir)

    # Download labels
    labels_path = data_root_dir / 'labels.npy'
    if not os.path.isfile(labels_path):
        raise DownloadError("Labels are missing, try to re-download them.")
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + '.npy')
        if not corruption_file_path.is_file():
            raise DownloadError(
                f"{corruption} file is missing, try to re-download it.")

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity *
                            n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])
    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]
    return x_test, y_test