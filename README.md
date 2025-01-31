# SNAP: Low-latency Test-Time Adaptation with Sparse Updates

## Getting Started

### Installation

1. Install packages.
    ```shell
    conda create --name snap python=3.7
    conda activate snap
    #CUDA=12.4
    pip install -r requirements.txt
    pip3 install torch torchvision torchaudio
    pip install git+https://github.com/RobustBench/robustbench.git
    ```
2. Modify `data_root` in `utils/config.py` pointing to the data root. -->

### Quick Example

Run SNAP with Tent on CIFAR10-C, Adaptation Rate 0.1.
```shell
# TENT (naive STTA)
python3 cta_eval.py --data=cifar10 --alg=tent --model=resnet18 --batch_size=16 --lr=1e-4 --device=cuda --workers=2 --test_corrupt=0 --eval_mode=continual --adaptrate=0.1 --mem_size=16 --alginf --adst=basic
# TENT + STTA + SNAP
python3 cta_eval.py --data=cifar10 --alg=tent --model=resnet18 --batch_size=16 --lr=1e-4 --device=cuda --workers=2 --test_corrupt=0 --eval_mode=continual --adaptrate=0.1 --mem_size=16 --alginf --adst=high_conf --rmst=WASS_OPP --memtype=pb --iobmn_k=1 --iobmn_s=1 --iobmn
```
