"""pytest conftest: mock torch if not available.

This runs before any test file is collected, so _engine.py's
`import torch` can succeed even in environments without torch
wheels (e.g. Python 3.14).
"""
import sys
import types
from unittest.mock import MagicMock

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    mod = types.ModuleType("torch")

    class MockDevice:
        def __init__(self, *args, **kwargs):
            pass
        def __repr__(self):
            return "cpu"

    mod.device = MockDevice
    mod.long = 0
    mod.float = 1
    mod.bfloat16 = 2
    mod.int64 = 3

    def mock_tensor(data, *args, **kwargs):
        import numpy as np
        return np.array(data, dtype=np.int64)
    mod.tensor = mock_tensor

    def manual_seed(s):
        pass
    mod.manual_seed = manual_seed

    class CudaMock:
        @staticmethod
        def is_available():
            return False
    mod.cuda = CudaMock

    class _Optim:
        def __init__(self, params, **kwargs):
            pass
        def zero_grad(self):
            pass
        def step(self):
            pass
        def state_dict(self):
            return {}
        def load_state_dict(self, d):
            pass

    class _OptimModule(types.ModuleType):
        def Adam(self, *a, **kw):
            return _Optim()
        def AdamW(self, *a, **kw):
            return _Optim()

    mod.optim = _OptimModule("torch.optim")
    mod.nn = MagicMock()

    sys.modules["torch"] = mod
    sys.modules["torch.nn"] = mod.nn
    sys.modules["torch.nn.functional"] = mod.nn.functional
    sys.modules["torch.optim"] = mod.optim
