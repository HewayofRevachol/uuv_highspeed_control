"""Compatibility entry point; implementation lives in training.ddat."""

import sys
from training import ddat as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    # Alias the module so legacy imports and mutable validation settings keep working.
    sys.modules[__name__] = _implementation
