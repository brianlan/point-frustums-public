import os
import torch


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
__version__ = "0.0.0"


# Import custom extensions
try:
    from point_frustums import _C
except ImportError as err:
    raise EnvironmentError(
        "The built C++/CUDA extensions could not be loaded. They should be present under `point_frustums/_C*.so` after "
        "installing the package."
    ) from err
