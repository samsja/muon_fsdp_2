# Muon FSDP2

muon implementation for FSDP2. Mainly copied [this gist](https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823) credits to [@main-horse](https://github.com/main-horse)

## Usage

```python
from muon_fsdp2 import Muon


optimizer = Muon([
    dict(
        params=model.square_params(),
        lr=2e-2,
        use_muon=True
    ),
    dict(
        params=model.non_square_params(),
        lr=3e-4,
        use_muon=False
    )
])

```
