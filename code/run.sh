#!/bin/bash
# File              : run.sh
# Author            : Zhepei Wang <zhepeiw03@gmail.com>
# Date              : 13.12.2019
# Last Modified Date: 13.12.2019
# Last Modified By  : Pezhin <cnicaaron@gmail.com>

export OMP_NUM_THREADS=1
source activate py369

# use wandb key from environment variable
wandb login $WANDB_API_KEY
# python main.py -bs 128 --num_workers 4 --lr 3e-4 --epochs 40 --seed 12 \
#     --vocab_size 8000 -M BLSTM -cad 0 \
#     --wandb_entity CAL --wandb_project cs547
python parallel_experiment_runner.py -cad 0 1 2 3 --epochs 15 -M BLSTM \
    --wandb_entity CAL --wandb_project cs547
source deactivate
unset OMP_NUM_THREADS
