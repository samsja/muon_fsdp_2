# Re-export everything from the nested module
from muon_fsdp2.muon_fsdp2 import __version__, Muon, MuonDDP

# Make sure all exports are available at the top level
__all__ = ["__version__", "Muon", "MuonDDP"]
