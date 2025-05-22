# SNAP: Low-latency Test-Time Adaptation with Sparse Updates

## Getting Started

### Requirements

1. Install packages & build environment using [conda](https://docs.conda.io/en/latest/). Please install [Anaconda](https://www.anaconda.com/) first.
    ```shell
    conda create --name snap python=3.7
    conda activate snap
    #CUDA=12.4
    pip install -r requirements.txt
    pip3 install torch torchvision torchaudio
    pip install git+https://github.com/RobustBench/robustbench.git
    ```
2. Prepare datasets: download at least one of the datasets listed below.
    - [CIFAR-10-C](https://zenodo.org/records/2535967)
    - [CIFAR-100-C](https://zenodo.org/records/3555552)
    - [ImageNet-C](https://zenodo.org/record/2235448)

3. Modify `data_root` in `utils/config.py` pointing to the data root path.

### Quick Example

Compare our *SNAP* against naive STTA (Sparse TTA) on CIFAR-10-C.

```shell
# naive STTA with Tent
python3 cta_eval.py --data=cifar10 --alg=tent --model=resnet18 --batch_size=16 --lr=1e-4 --device=cuda --workers=2 --test_corrupt=0 --eval_mode=continual --adaptrate=0.1 --mem_size=16 --alginf --adst=basic
# SNAP with Tent
python3 cta_eval.py --data=cifar10 --alg=tent --model=resnet18 --batch_size=16 --lr=1e-4 --device=cuda --workers=2 --test_corrupt=0 --eval_mode=continual --adaptrate=0.1 --mem_size=16 --alginf --adst=high_conf --rmst=WASS_OPP --memtype=pb --iobmn_k=1 --iobmn_s=1 --iobmn
```

### Prepare Source Model (pre-trained on clean image)
The source model refers to a model trained exclusively on clean (uncorrupted) data.

1. We provide pre-trained source model (ResNet18) of CIFAR-10 and CIFAR-100 in the `data/robustbench_models` directory of this repository.

2. For ImageNet, we used pre-trained weights (ResNet50 and ViT-B) from [TorchVision](https://docs.pytorch.org/vision/main/models.html), which are automatically downloaded by the provided script.

## How to run the evlauation

### Evaluation SNAP with Test-Time Adpatation

The main evaluation script is provided to compare the adaptation performance with and without SNAP across various adaptation rates. The example script below runs evaluations based on Tent for all datasets (CIFAR-10-C, CIFAR-100-C, and ImageNet-C) and generates evaluation logs for adaptation rates ranging from 0.01 to 0.5.

```shell
# Run Tent+SNAP vs Tent+naiveSTTA
./test_script/tent/run_tent.sh
```

Similar scripts for other algorithms (CoTTA, EATA, SAR, and RoTTA) are also available in the `test_script` directory.

<!-- ### Efficiency Comparision Result

#### Accuracy and Latency Evaluation With and Without SNAP (Adaptation Rate: 0.1, Dataset: CIFAR-100-C)

| TTA methods        | Accuracy (%)  | Latency per batch (s) |
| ------------------ |---------------- | -------------- |
| Tent  |     55.76         |      4.54      |
| +STTA  |     52.84         |      3.34      |
|  +SNAP  |     55.84         |      3.67      |
| CoTTA  |     49.39         |      74.77     |
| +STTA  |     35.85         |      4.94      |
|  +SNAP  |     50.52         |      4.95      | -->

### Logs

Parse the textual log files and generate a CSV summary.

```shell
python3 parse_log.py
```

Make sure to update the `parent_txtlog_folder` path inside `parse_log.py` to point to the correct directory containing your log files.

### Tested Environment

The following environment was used for testing and evaluation reported on the paper:

- **OS**: Ubuntu 22.04.2 LTS (Jammy Jellyfish)
- **GPU**: NVIDIA GeForce RTX 3090
- **GPU Driver Version**: 550.144.03
- **CUDA Version**: 12.4 (nvcc 12.4.131)
- **GCC Version**: 12.3.0

You may experience compatibility issues with different driver/CUDA versions. Please ensure consistency with this tested setup where possible.

