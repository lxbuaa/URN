import random

import os

import numpy as np
import torch


def set_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True