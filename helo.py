from muon_fsdp2 import Muon


optimizer = Muon(
    [dict(params=model.square_params(), lr=2e-2, use_muon=True), dict(params=model.non_square_params(), lr=2e-2, use_muon=False)]
)
