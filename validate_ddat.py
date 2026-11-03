"""Compatibility entry point; implementation lives in evaluation.ddat."""

import sys
from evaluation import ddat as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    # Alias the module so legacy imports and mutable validation settings keep working.
    sys.modules[__name__] = _implementation
