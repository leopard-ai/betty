#!/usr/bin/env bash

# Train baseline with 100% data (we follow DynaMS paper for setting our hyper-parameters)

WORLD_SIZE=4 torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port 47769 main.py --data_dir imagenet/imagenet.hdf5 --batch_size 64 --checkpoint_directory "output/resnet50_64_4_120_nesterov_wd1e-4_wojit_seed42" --num_workers 4 --layers 50 --strategy "distributed" --weight_decay 1e-4 --nesterov --baseline --seed 42

# Meta learning the weighing network

WORLD_SIZE=4 torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port 47769 main.py --data_dir imagenet/imagenet.hdf5 --batch_size 64 --checkpoint_directory "output_reweight/run1_resnet50" --num_workers 4 --layers 50 --strategy "distributed" --weight_decay 1e-4 --nesterov --seed 42

# Using the meta learned weighing network to compute weights for all ImageNet-1K training examples

python filter.py --data-dir imagenet/imagenet.hdf5 --checkpoint_directory "output_reweight/run1_resnet50" --layers 50 --batch_size 512 --desc "130_150k" > logs/filter_run1_resnet50_130_150k.out

# Train ResNet-50 with 90% data...(prune 10% examples based upon meta learned weighing network)

WORLD_SIZE=4 torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port 47769 main.py --data_dir imagenet/imagenet.hdf5 --batch_size 64 --checkpoint_directory "output_prune/resnet50_64_4_120_nesterov_wd1e-4_wojit/prune0.9_130_150k_seed42" --num_workers 4 --layers 50 --strategy "distributed" --weight_decay 1e-4 --nesterov --baseline --seed 42 --prune --frac_data_kept 0.9 --instance_weights_dir "output_reweight/run1_resnet50/130_150k" --prune_strategy "metaweight"

# Random weighing for all ImageNet-1K training examples

python filter.py --data-dir imagenet/imagenet.hdf5 --checkpoint_directory "output_reweight/run1_resnet50" --layers 50 --batch_size 512 --desc "randomprune" --seed 42 --random > logs/filter_run1_resnet50_random.out

# Train ResNet-50 with 90% data...(randomly prune 10% examples)

WORLD_SIZE=4 torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port 47769 main.py --data_dir imagenet/imagenet.hdf5 --batch_size 64 --checkpoint_directory "output_prune/resnet50_64_4_120_nesterov_wd1e-4_wojit/randomprune0.9_seed42" --num_workers 4 --layers 50 --strategy "distributed" --weight_decay 1e-4 --nesterov --baseline --seed 42 --prune --frac_data_kept 0.9 --instance_weights_dir "output_reweight/run1_resnet50/randomprune" --prune_strategy "random"