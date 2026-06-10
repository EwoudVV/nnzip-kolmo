"""Mock torch for environments without installable torch wheels."""
import sys
import types
from unittest.mock import MagicMock

mod = types.ModuleType("torch")


class MockDevice:
    def __init__(self, *a, **kw): pass
    def __repr__(self): return "cpu"


mod.device = MockDevice
mod.long = 0
mod.float = 1
mod.bfloat16 = 2
mod.int64 = 3


def mock_tensor(data, *a, **kw):
    import numpy as np
    return np.array(data, dtype=np.int64)


mod.tensor = mock_tensor
mod.manual_seed = lambda s: None


class CudaMock:
    is_available = staticmethod(lambda: False)


mod.cuda = CudaMock


class _Optim:
    def __init__(self, p, **kw): pass
    def zero_grad(self): pass
    def step(self): pass
    def state_dict(self): return {}
    def load_state_dict(self, d): pass


class _OptimModule(types.ModuleType):
    def Adam(self, *a, **kw): return _Optim()
    def AdamW(self, *a, **kw): return _Optim()


mod.optim = _OptimModule("torch.optim")
mod.nn = MagicMock()

sys.modules["torch"] = mod
sys.modules["torch.nn"] = mod.nn
sys.modules["torch.nn.functional"] = mod.nn.functional
sys.modules["torch.optim"] = mod.optim
