# Muon FSDP2

muon implementation for FSDP2. Mainly copied [this gist](https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823) credits to [@main-horse](https://github.com/main-horse)

## Usage

```python
from muon_fsdp2 import Muon

optimizer = Muon(model.parameters(), lr=0.001)
```
