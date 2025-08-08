# Muon fsdp 2

This codebase implements the muon optimizer compatible with fsdp2 as described in [this blog post](https://main-horse.github.io/posts/parallelizing-muon/)

Most of the important code was developed by main-horse [here](https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823)
This repo integrates that code into a training codebase and optimizes the communication part (we do gather scatter instead of all_gather)


This repo is composed of two parts:

- `src/muon_fsdp2` is the implementation of the muon optimizer compatible with fsdp2
- `src/zeroband` is the training codebase

## Muon FSDP2 package

this is a standalone package that can be used to train models with the muon optimizer. 

install the package from pypi

```bash
uv pip install muon-fsdp2
```

or from source

```bash
uv pip install git+https://github.com/samsja/muon_fsdp_2.git@main
```


example usage

```python
from muon_fsdp2 import Muon


optimizer = Muon([
    dict(
        params=model.square_params(),
        lr=1e-3,
        use_muon=True
    ),
    dict(
        params=model.non_square_params(),
        lr=1e-3,
        use_muon=False
    )
]) 

```

## ZeroBand

ZeroBand is a fork of [this repo](https://github.com/PrimeIntellect-ai/prime), a standalone training codebase for LLMs specifically designed for using fsdp2 and muon optimizer.


## Install

```bash
export HF_TOKEN=
```

then use the default install script to install the dependencies

```
curl -sSL https://raw.githubusercontent.com/samsja/muon_fsdp_2/main/install.sh | bash
source $HOME/.local/bin/env

```

or do it manually

```bash
git clone https://github.com/samsja/muon_fsdp_2
cd muon_fsdp_2
uv sync
```

run debug

```bash
PRIME_DEBUG=1 uv  run torchrun --nproc_per_node=2 train_fsdp.py  @ configs/debug/normal.toml
```

run 150

```bash
uv  run torchrun --nproc_per_node=8 train_fsdp.py @ configs/150M/H100.toml
```

run 1b 

```bash
uv  run torchrun --nproc_per_node=8 train_fsdp.py @ configs/1B/H100.toml
```

run 7b

```bash
uv  run torchrun --nproc_per_node=8 train_fsdp.py @ configs/7B/H100.toml
```

### benchmark
old code not updated yet

| Model Size | GPUs | GPU Type | MFU |
|------------|------|----------|-----|
| 1B         | 8    | H100 sxm | 45% |
| 7B         | 8    | H100 sxm | 49% |


## convergence 150M

old code not updated yet

![Screenshot from 2025-03-16 21-38-16](https://github.com/user-attachments/assets/5b93ec21-3e71-4f66-be47-7e07bc88c77e)


to reproduce the convergence, run
```bash
uv  run torchrun --nproc_per_node=8 train_fsdp.py @ configs/150M/H100.toml
uv  run torchrun --nproc_per_node=8 train_ddp.py @ configs/150M/H100.toml
```
