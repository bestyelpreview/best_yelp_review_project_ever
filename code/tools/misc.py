#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : misc.py
# Author            : Zhepei Wang <zhepeiw03@gmail.com>
# Date              : 27.11.2019
# Last Modified Date: 27.11.2019
# Last Modified By  : Zhepei Wang <zhepeiw03@gmail.com>

import torch
import numpy as np
import random

def reset_seed(seed=1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
