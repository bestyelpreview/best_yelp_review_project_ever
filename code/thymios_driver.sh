# @brief Setting up the experiment for wandb reports
# @author Efthymios Tzinis {etzinis2@illinois.edu}
# @copyright University of illinois at Urbana Champaign

python3 -m wandb login d5edbf3b5a4718b11cdf5f180fd7e58a6e012b43
python3 main.py --batch_size 16 --num_workers 4 --lr 3e-4 --epochs 100 \
    --seed 12 --vocab_size 8000 --wandb_entity CAL --wandb_project cs547