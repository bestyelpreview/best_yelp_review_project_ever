#!/bin/bash
# File              : zhepei_driver.sh
# Author            : Zhepei Wang <zhepeiw03@gmail.com>
# Date              : 26.11.2019
# Last Modified Date: 26.11.2019
# Last Modified By  : Zhepei Wang <zhepeiw03@gmail.com>

source activate py369
python main.py --batch_size 64 --num_workers 4 --lr 3e-4 --epochs 100
source deactivate
