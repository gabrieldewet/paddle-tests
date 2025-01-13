import platform

import torch

N_CPU = 12
USE_GPU = False


if platform.system() == "Darwin":
    PLATFORM = "mac"
elif USE_GPU and torch.cuda.is_available():
    PLATFORM = "gpu"
else:
    PLATFORM = "cpu"
