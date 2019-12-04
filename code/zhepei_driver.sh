#!/bin/bash
# File              : zhepei_driver.sh
# Author            : Zhepei Wang <zhepeiw03@gmail.com>
# Date              : 26.11.2019
# Last Modified Date: 26.11.2019
# Last Modified By  : Zhepei Wang <zhepeiw03@gmail.com>

export OMP_NUM_THREADS=1
source activate py369
wandb login e6fd3c9f449fd9cda1a86b60c7775c5700b11f98
python main.py --batch_size 128 --num_workers 4 --lr 3e-4 --epochs 100 --seed 12 \
    --vocab_size 8000 \
    --wandb_entity CAL --wandb_project cs547
source deactivate
unset OMP_NUM_THREADS
