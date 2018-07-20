#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/cifar10/main.py \
  --data_format="NCHW" \
  --search_for="macro" \
  --reset_output_dir \
  --data_path="data/cifar10" \
  --output_dir="outputs" \
  --batch_size=128 \
  --num_epochs=310 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_use_aux_heads \
  --child_num_layers=12 \
  --child_out_filters=36 \
  --child_l2_reg=0.00025 \
  --child_num_branches=6 \
  --child_num_cell_layers=5 \
  --child_keep_prob=0.90 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr_cosine \
  --child_lr_max=0.05 \
  --child_lr_min=0.0005 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --controller_training \
  --controller_search_whole_channels \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_sync_replicas \
  --controller_num_aggregate=20 \
  --controller_train_steps=50 \
  --controller_lr=0.001 \
  --controller_tanh_constant=1.5 \
  --controller_op_tanh_reduce=2.5 \
  --controller_skip_target=0.4 \
  --controller_skip_weight=0.8 \
  "$@"

  # --data_format="NCHW" \ - format of the data
  # --search_for="macro" - macro search for architecture or micro search for cells
  # --reset_output_dir \ - whether to clear outputs dir if there is something inside
  # --data_path="data/cifar10" \ - path to images
  # --output_dir="outputs" \ - path to output dir (for model checkpoints, tf events)
  # --batch_size=128 \ - training batch size
  # --num_epochs=310 \ - number of epochs to train for
  # --log_every=50 \ - Every how many steps do we log lr, acc, loss etc
  # --eval_every_epochs=1 \ Evaluate accuracy every how many epochs
  # --child_use_aux_heads \ Whether to use auxiliary loss
  # --child_num_layers=12 \ Number of layers for the architecture
  # --child_out_filters=36 \ Number of filters for output channels
  # --child_l2_reg=0.00025 \ l2 regularisation
  # --child_num_branches=6 \ number of branches to explore
  # --child_num_cell_layers=5 \ 
  # --child_keep_prob=0.90 \ Determine keep ratio (1 - dropout)
  # --child_drop_path_keep_prob=0.60 \
  # --child_lr_cosine \ Cosine annealing policy for LR
  # --child_lr_max=0.05 \ Restart point after each cycle
  # --child_lr_min=0.0005 \ The lowest the lr can go
  # --child_lr_T_0=10 \ Initial number of iterations for the cycle
  # --child_lr_T_mul=2 \ How much longer should the next iterations cycle be
  # --controller_training \
  # --controller_search_whole_channels \
  # --controller_entropy_weight=0.0001 \
  # --controller_train_every=1 \
  # --controller_sync_replicas \
  # --controller_num_aggregate=20 \
  # --controller_train_steps=50 \
  # --controller_lr=0.001 \
  # --controller_tanh_constant=1.5 \
  # --controller_op_tanh_reduce=2.5 \
  # --controller_skip_target=0.4 \
  # --controller_skip_weight=0.8 \