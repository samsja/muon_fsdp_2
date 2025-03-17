# Muon fsdp 2

This codebase implement muon optimizer compatbile with fsdp2 as described in [this blog post](https://main-horse.github.io/posts/parallelizing-muon/)

Most of the important code has been developed by main-horse [here](https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823)
This repo add the code into a training codebase and optimizer the communication part (do gather scatter instead of all_gather)

## Install

```bash
export HF_TOKEN=
```

then

```
curl -sSL https://raw.githubusercontent.com/samsja/muon_fsdp_2/main/install.sh | bash
source $HOME/.local/bin/env

```

run debug

```bash
uv  run torchrun --nproc_per_node=2 train_fsdp.py  @ configs/debug/normal.toml
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

| Model Size | GPUs | GPU Type | MFU |
|------------|------|----------|-----|
| 1B         | 8    | H100 sxm | 45% |
| 7B         | 8    | H100 sxm | 38% |


## convergence 150M

![Screenshot from 2025-03-16 21-38-16](https://github.com/user-attachments/assets/5b93ec21-3e71-4f66-be47-7e07bc88c77e)


to reproduce the convergence, run
```bash
uv  run torchrun --nproc_per_node=8 train_fsdp.py @ configs/150M/H100.toml
uv  run torchrun --nproc_per_node=8 train_ddp.py @ configs/150M/H100.toml
```
