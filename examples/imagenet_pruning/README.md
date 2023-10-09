# ImageNet Data Pruning

### Download ImageNet dataset
```bash
bash extract_ILSVRC.sh
```

### ImageNet data pruning
**1. (Meta-)learn data importance weights**
```bash
WORLD_SIZE=4 torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port 47769 main.py --data_dir imagenet/imagenet.hdf5 --batch_size 64 --checkpoint_directory "output_reweight/run1_resnet50" --num_workers 4 --layers 50 --strategy "distributed" --weight_decay 1e-4 --nesterov
```

**2. Sort data by importance weights**
```bash
python filter.py --data-dir imagenet/imagenet.hdf5 --checkpoint_directory "output_reweight/run1_resnet50" --layers 50 --batch_size 512 --desc "130_150k" > logs/filter_run1_resnet50_130_150k.out
```

**3. Train with pruned data**
```bash
WORLD_SIZE=4 torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port 47769 main.py --data_dir imagenet/imagenet.hdf5 --batch_size 64 --checkpoint_directory "output_prune/resnet50_64_4_120_nesterov_wd1e-4_wojit/prune0.9_130_150k_seed42" --num_workers 4 --layers 50 --strategy "distributed" --weight_decay 1e-4 --nesterov --baseline --seed 42 --prune --frac_data_kept 0.9 --instance_weights_dir "output_reweight/run1_resnet50/130_150k" --prune_strategy "metaweight"
```

Or, you can simply run `bash paper_runs.sh` to get all results.
