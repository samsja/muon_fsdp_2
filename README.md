# Muon fsdp 2

This codebase implement muon optimizer compatbile with fsdp2 as described in [this blog post](https://main-horse.github.io/posts/parallelizing-muon/)

Original optimizer code credits to main-horse [here](https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823)

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
| 7B         | 8    | H100 sxm | n/a |


## convergence








